import json
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =========================
# CONFIG (edit if needed)
# =========================
INPUT_DIR = r"data/knowledge_base/preprocessed_markdown"
OUTPUT_DIR = r"data/.cache/integrated_knowledge_maps"

# Output for embedding (JSONL: {id, text, metadata})
OUTPUT_NODES_JSONL = r"data/.cache/integrated_knowledge_maps_nodes.jsonl"

# Embedding node sizing (split large clusters into multiple nodes)
EMBED_NODE_MAX_CHARS = 4000
EMBED_NODE_MIN_CHARS = 800

# LLM behavior (uses your existing `.env` config via app.core.llama.init_llm_from_env)
USE_LLM = True
LLM_STRATEGY = "per_entity"  # "per_entity" (recommended) or "single_call"
LLM_REQUEST_JSON_MODE = True
LLM_MAX_TOKENS = 1200
LLM_MAX_CHARS_PER_ENTITY = 12000
LLM_MAX_TOTAL_CHARS = 36000
LLM_RETRY_MAX = 5
LLM_RETRY_INITIAL_DELAY_S = 2.0
LLM_RETRY_BACKOFF = 2.0

# Heuristic behavior (used both for entity discovery and as fallback if LLM disabled/fails)
MAX_ENTITIES_PER_DOC = 24
TOP_SECTIONS_PER_ENTITY = 6
SECTION_MIN_CHARS = 80

# Progress logging
PROGRESS_LOG = True
PROGRESS_LOG_EVERY_ENTITY = True


# -------------------------
# Utilities
# -------------------------


def _ensure_agent_env() -> None:
    """
    If user runs in an env without llama-index, re-exec with conda env `agent`.
    """
    if os.getenv("IKM_NO_REEXEC"):
        return
    try:
        import llama_index  # type: ignore  # noqa: F401

        return
    except Exception:
        pass

    cmd = ["conda", "run", "-n", "agent", "python", str(Path(__file__).resolve())]
    env = dict(os.environ)
    env["IKM_NO_REEXEC"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    print("Re-running with: " + " ".join(cmd))
    try:
        r = subprocess.run(cmd, env=env)
        raise SystemExit(r.returncode)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Cannot find `conda` to re-exec into env `agent`. "
            "Run inside the correct env:\n"
            "  conda run -n agent python scripts/build_integrated_knowledge_maps.py"
        ) from e


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    # Vietnamese "đ/Đ" does not decompose under NFD, so normalize it manually.
    return s.replace("đ", "d").replace("Đ", "D")


def _norm(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = _strip_accents(s).lower()
    return s


def _slugify(s: str) -> str:
    s2 = _norm(s)
    s2 = re.sub(r"[^a-z0-9]+", "_", s2).strip("_")
    return s2


def _extract_json(text: str) -> Any:
    s = (text or "").strip()
    if not s:
        raise ValueError("LLM returned empty output.")
    if "```" in s:
        parts = s.split("```")
        for i in range(len(parts) - 1):
            block = parts[i + 1].strip()
            if block.lower().startswith("json"):
                block = block[4:].strip()
            if block.startswith("{") or block.startswith("["):
                s = block
                break
    first_candidates = [i for i in (s.find("["), s.find("{")) if i != -1]
    if not first_candidates:
        raise ValueError("No JSON object/array found in LLM output.")
    start = min(first_candidates)
    s2 = s[start:].lstrip()
    decoder = json.JSONDecoder()
    obj, _end = decoder.raw_decode(s2)
    return obj


def _derive_tenant_from_filename(path: Path) -> str:
    name = path.stem
    if name.lower().startswith("tenant_"):
        name = name[len("tenant_") :]
    safe = re.sub(r"[^a-z0-9_]+", "", name.lower())
    return safe or name.lower()


@dataclass(frozen=True)
class Section:
    level: int
    heading: str
    body: str

    @property
    def heading_path(self) -> str:
        return self.heading.strip() or "Thông tin chung"


def parse_markdown_sections(md_text: str) -> List[Section]:
    """
    Minimal markdown heading parser: groups content under headings (#..######).
    """
    text = (md_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()
    sections: List[Section] = []
    cur_level = 1
    cur_heading = "Thông tin chung"
    cur_lines: List[str] = []

    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def flush():
        nonlocal cur_lines, cur_heading, cur_level
        body = "\n".join(cur_lines).strip()
        if body:
            sections.append(Section(level=cur_level, heading=cur_heading, body=body))
        cur_lines = []

    for ln in lines:
        m = heading_re.match(ln)
        if m:
            flush()
            cur_level = len(m.group(1))
            cur_heading = m.group(2).strip()
            cur_lines = []
        else:
            cur_lines.append(ln)
    flush()

    # Filter empty/noise
    out = []
    for s in sections:
        if len(s.body.strip()) >= SECTION_MIN_CHARS or s.heading.strip():
            out.append(s)
    return out


# -------------------------
# Heuristic entity discovery
# -------------------------


GENERIC_HEADINGS = {
    "hoc phi",
    "phi",
    "phi tai lieu",
    "lich khai giang",
    "lich hoc",
    "khai giang",
    "chinh sach",
    "quy dinh",
    "noi quy",
    "cam ket",
    "lien he",
    "gio hoat dong",
    "thong tin lien he",
    "thong tin lien he chung",
    "khung gio hoat dong",
    "doi ngu",
    "giao vien",
    "tai lieu",
    "uu dai",
    "hoc bong",
    "hoan phi",
    "bao luu",
    "muc luc",
    "gioi thieu",
    "thong tin chung",
}

ENTITY_HINT_KEYWORDS = [
    "ielts",
    "toeic",
    "cefr",
    "cambridge",
    "starters",
    "movers",
    "flyers",
    "ket",
    "pet",
    "giao tiep",
    "business english",
    "thieu nhi",
    "mau giao",
    "tieng anh thieu nhi",
    "tieng anh thieu nien",
    "lop",
    "khoa",
    "chuong trinh",
    "khoa hoc",
]


def _looks_like_entity_name(s: str) -> bool:
    # Strip common numbering prefixes: "1.", "1.1", "I.", "II.", "5)"
    s2 = re.sub(r"^\s*(\d+(\.\d+){0,3}|[ivxlcdm]{1,8})\s*[\)\.\-:]\s*", "", s.strip(), flags=re.IGNORECASE)
    n = _norm(s2)
    if not n or len(n) < 3:
        return False
    if n in GENERIC_HEADINGS:
        return False
    # Contains hints or looks like a named program
    if any(kw in n for kw in ENTITY_HINT_KEYWORDS):
        return True
    # Short title-case-ish line
    if len(n.split()) <= 8 and any(ch.isalpha() for ch in n):
        return True
    return False


def discover_entities(sections: List[Section]) -> List[Tuple[str, str]]:
    """
    Return list of (entity_id, entity_name) using simple heuristics.
    """
    candidates: Dict[str, str] = {}

    # 1) Headings
    for s in sections:
        h = s.heading.strip()
        if not h:
            continue
        if _looks_like_entity_name(h):
            clean_h = re.sub(r"^\s*(\d+(\.\d+){0,3}|[ivxlcdm]{1,8})\s*[\)\.\-:]\s*", "", h, flags=re.IGNORECASE).strip()
            eid = _slugify(clean_h)
            if eid and eid not in GENERIC_HEADINGS:
                candidates.setdefault(eid, clean_h)

    # 2) Inline patterns in bodies (e.g., "Khóa IELTS ...", "Lớp TOEIC ...")
    # Keep this permissive (Unicode-safe) and rely on `_looks_like_entity_name` to filter.
    body_re = re.compile(r"(?im)\b(khoa\s*hoc|chuong\s*trinh|lop)\s+([^\n]{3,80})$")
    for s in sections:
        for m in body_re.finditer(s.body):
            name = (m.group(2) or "").strip()
            if not name:
                continue
            if _looks_like_entity_name(name):
                eid = _slugify(name)
                if eid and eid not in GENERIC_HEADINGS:
                    candidates.setdefault(eid, name)

    # 3) Tuition-style lines: "IELTS Foundation: 8.500.000 VND / 4 tháng"
    # Capture "Name: ..." where Name can contain Unicode (Vietnamese).
    tuition_line = re.compile(r"(?m)^\s*[-*]?\s*([^\n:]{3,80})\s*:\s+\S.+$")
    for s in sections:
        for m in tuition_line.finditer(s.body):
            name = (m.group(1) or "").strip()
            if not name:
                continue
            if not _looks_like_entity_name(name):
                continue
            n = _norm(name)
            if n in GENERIC_HEADINGS:
                continue
            eid = _slugify(name)
            if eid and eid not in GENERIC_HEADINGS:
                candidates.setdefault(eid, name)

    entities = [(eid, name) for eid, name in candidates.items()]
    entities.sort(key=lambda x: x[0])
    return entities[:MAX_ENTITIES_PER_DOC]


# -------------------------
# Linkage: collect relevant sections for an entity
# -------------------------


def _token_set(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", _norm(s)) if t}


def score_section_for_entity(entity_name: str, section: Section) -> float:
    """
    Score relevance of a section to an entity using token overlap + heading boosts.
    """
    ent_tokens = _token_set(entity_name)
    if not ent_tokens:
        return 0.0
    h_tokens = _token_set(section.heading)
    b_tokens = _token_set(section.body[:2000])
    overlap_h = len(ent_tokens.intersection(h_tokens))
    overlap_b = len(ent_tokens.intersection(b_tokens))
    score = 3.0 * overlap_h + 1.0 * overlap_b

    # Boost if exact string appears
    en = _norm(entity_name)
    if en and en in _norm(section.heading):
        score += 6.0
    if en and en in _norm(section.body[:3000]):
        score += 2.0

    # Encourage finance/schedule/policy sections (often shared but still useful),
    # but only when we see at least *some* evidence of relevance in the body.
    hn = _norm(section.heading)
    if overlap_b > 0:
        if any(k in hn for k in ["hoc phi", "phi", "uu dai"]):
            score += 1.0
        if any(k in hn for k in ["lich", "khai giang", "gio hoc"]):
            score += 1.0
        if any(k in hn for k in ["chinh sach", "quy dinh", "cam ket", "bao luu", "hoan phi"]):
            score += 1.0
    return float(score)


def collect_entity_sections(sections: List[Section], entity_name: str) -> List[Section]:
    scored: List[Tuple[float, Section]] = []
    for s in sections:
        sc = score_section_for_entity(entity_name, s)
        if sc > 0:
            scored.append((sc, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _sc, s in scored[:TOP_SECTIONS_PER_ENTITY]]


def _render_sections(sections: List[Section]) -> str:
    parts: List[str] = []
    for s in sections:
        parts.append(f"[{s.heading_path}]\n{s.body}")
    return "\n\n---\n\n".join(parts).strip()


def _section_kind(heading: str) -> str:
    h = _norm(heading)
    if any(k in h for k in ["hoc phi", "phi", "uu dai", "hoc bong"]):
        return "tuition"
    if any(k in h for k in ["lich", "khai giang", "gio hoc", "thoi gian hoc"]):
        return "schedule"
    if any(k in h for k in ["chinh sach", "quy dinh", "noi quy", "cam ket", "bao luu", "hoan phi"]):
        return "policy"
    return "general"


def _extract_relevant_lines(text: str, entity_name: str, *, max_lines: int = 16) -> str:
    """
    Reduce redundant tables by keeping only lines likely relevant to the entity.
    """
    lines = (text or "").splitlines()
    if not lines:
        return ""

    ent_tokens = _token_set(entity_name)
    keep: List[str] = []
    used = [False] * len(lines)

    def mark(i: int) -> None:
        if 0 <= i < len(lines) and not used[i]:
            used[i] = True
            keep.append(lines[i])

    # 1) Keep lines mentioning the entity (and a small window).
    for i, ln in enumerate(lines):
        nln = _norm(ln)
        if not nln.strip():
            continue
        if ent_tokens and any(t in nln for t in ent_tokens):
            for j in range(i, min(i + 3, len(lines))):
                mark(j)

    # 2) If nothing matched, keep a few money/duration lines near the top.
    if not keep:
        money_or_time = re.compile(r"(?i)(₫|vnd|dong|trieu|tr|thang|tháng|tuan|tuần|ielts|toeic|a1|a2|b1|b2|c1|c2)")
        for i, ln in enumerate(lines):
            if money_or_time.search(ln):
                mark(i)
            if len(keep) >= max_lines:
                break

    # 3) Hard cap.
    out = "\n".join([ln for ln in keep if ln.strip()]).strip()
    if len(out) > 2200:
        out = out[:2200].rstrip() + "\n...[TRUNCATED]..."
    return out


def _render_entity_context(entity_name: str, picked_sections: List[Section]) -> str:
    """
    Render sections for one entity with de-duplication: tuition/policy/schedule sections are compacted.
    """
    parts: List[str] = []
    for s in picked_sections:
        body = s.body
        if _section_kind(s.heading) in ("tuition", "policy", "schedule"):
            body = _extract_relevant_lines(body, entity_name)
        parts.append(f"[{s.heading_path}]\n{(body or '').strip()}")
    return "\n\n---\n\n".join([p for p in parts if p.strip()]).strip()


def _supplement_generic_sections(sections: List[Section], entity_name: str, picked: List[Section]) -> List[Section]:
    """
    Ensure key operational info (tuition/schedule/policy) is present without copying whole tables.
    Adds compacted snippets only when the picked set doesn't already contain that kind.
    """
    present = {_section_kind(s.heading) for s in picked}
    out = list(picked)

    def build(kind: str, label: str) -> Optional[Section]:
        snippets: List[str] = []
        for s in sections:
            if _section_kind(s.heading) != kind:
                continue
            snip = _extract_relevant_lines(s.body, entity_name, max_lines=14)
            if snip.strip():
                snippets.append(snip.strip())
        if not snippets:
            return None
        merged = "\n".join(snippets).strip()
        if len(merged) > 2200:
            merged = merged[:2200].rstrip() + "\n...[TRUNCATED]..."
        return Section(level=6, heading=f"{label} (trích lọc)", body=merged)

    if "tuition" not in present:
        s = build("tuition", "Học phí")
        if s:
            out.append(s)
    if "schedule" not in present:
        s = build("schedule", "Lịch học / Khai giảng")
        if s:
            out.append(s)
    if "policy" not in present:
        s = build("policy", "Chính sách / Cam kết")
        if s:
            out.append(s)

    return out


def collect_entity_context(sections: List[Section], entity_name: str) -> str:
    picked = collect_entity_sections(sections, entity_name)
    picked2 = _supplement_generic_sections(sections, entity_name, picked)
    return _render_entity_context(entity_name, picked2)


def collect_general_context(sections: List[Section]) -> str:
    """
    Non-course/org-level info bucket.
    """
    return _render_sections(sections[: min(10, len(sections))])


# -------------------------
# LLM synthesis
# -------------------------


def _call_llm(prompt: str) -> str:
    from llama_index.core import Settings

    llm = getattr(Settings, "_llm", None)
    if llm is None:
        from app.core.llama import init_llm_from_env

        init_llm_from_env()
        llm = getattr(Settings, "_llm", None)
    if llm is None:
        raise RuntimeError("LLM is not initialized. Check `.env` LLM_PROVIDER + API keys.")

    delay = float(LLM_RETRY_INITIAL_DELAY_S)
    last_err: Exception | None = None
    for attempt in range(1, int(LLM_RETRY_MAX) + 1):
        try:
            kwargs: Dict[str, object] = {"max_tokens": int(LLM_MAX_TOKENS)}
            if LLM_REQUEST_JSON_MODE:
                kwargs["temperature"] = 0.0
                kwargs["response_format"] = {"type": "json_object"}

            try:
                resp = llm.complete(prompt, **kwargs)
            except TypeError:
                # Some LLM wrappers may not accept extra kwargs.
                resp = llm.complete(prompt)
            return getattr(resp, "text", None) or str(resp)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            retryable = any(k in msg for k in ["timeout", "timed out", "429", "rate", "quota", "temporarily", "connection"])
            if attempt >= int(LLM_RETRY_MAX) or not retryable:
                break
            time.sleep(delay + random.uniform(0.0, 0.6))
            delay *= float(LLM_RETRY_BACKOFF)
    raise RuntimeError(f"LLM call failed after {LLM_RETRY_MAX} attempts: {last_err}") from last_err


def _apply_budget_per_entity(contexts: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    contexts: list of (entity_id, entity_name, context_text)
    """
    out = []
    for eid, name, ctx in contexts:
        ctx2 = (ctx or "").strip()
        if len(ctx2) > LLM_MAX_CHARS_PER_ENTITY:
            ctx2 = ctx2[:LLM_MAX_CHARS_PER_ENTITY].rstrip() + "\n...[TRUNCATED]..."
        out.append((eid, name, ctx2))
    return out


def _apply_total_budget(block: str) -> str:
    s = (block or "").strip()
    if len(s) <= LLM_MAX_TOTAL_CHARS:
        return s
    return s[:LLM_MAX_TOTAL_CHARS].rstrip() + "\n...[TRUNCATED]..."


def build_prompt(*, tenant_id: str, source_file: str, entity_contexts: List[Tuple[str, str, str]], general_context: str) -> str:
    entities_list = "\n".join([f"- {eid}: {name}" for eid, name, _ in entity_contexts]) or "- (none)"

    ctx_blocks = []
    for eid, name, ctx in entity_contexts:
        if not ctx.strip():
            continue
        ctx_blocks.append(f"### ENTITY {eid} ({name})\n{ctx}")

    prompt_body = "\n\n".join(ctx_blocks)
    prompt_body = _apply_total_budget(prompt_body)

    general_block = general_context.strip()
    if general_block:
        general_block = _apply_total_budget(general_block[:LLM_MAX_CHARS_PER_ENTITY])

    return (
        "Bạn là một Document Architect chuyên nghiệp. Nhiệm vụ: tạo Integrated Knowledge Maps cho hệ thống RAG SaaS.\n"
        "Ràng buộc:\n"
        "- CHỈ dùng thông tin trong văn bản được cung cấp.\n"
        "- Không bịa đặt.\n"
        "- Nếu thông tin không gắn với khóa học/dịch vụ cụ thể, gộp vào entity \"thong_tin_chung_ve_to_chuc\".\n"
        "- Output CHỈ là JSON ARRAY hợp lệ, không markdown, không giải thích.\n"
        "\n"
        "Yêu cầu cho kiến thức (knowledge_cluster):\n"
        "- Hãy viết như một chuyên gia tư vấn đang trả lời khách hàng.\n"
        "- Kết nối các mốc thời gian, tiền bạc và cam kết thực tế vào câu văn.\n"
        "- Ví dụ: \"Khóa học IELTS 6.5 học trong 4 tháng với học phí 10tr, cam kết học lại nếu trượt...\"\n"
        "- Không copy nguyên bảng học phí; nếu cần thì trích 1-3 dòng liên quan.\n"
        "\n"
        "Yêu cầu cho metadata.evidence (dictionary):\n"
        "- evidence là JSON OBJECT (không phải list).\n"
        "- Chỉ dùng các giá trị trích xuất được từ tài liệu.\n"
        "- Schema gợi ý:\n"
        "  {\"duration_months\": [number], \"duration_weeks\": [number], \"tuition_vnd\": [number], \"ielts_target\": [number], \"toeic_target\": [number], \"cefr_target\": [string]}\n"
        "\n"
        "Output format (JSON ARRAY): mỗi object:\n"
        "{\n"
        "  \"entity_id\": \"ten_thuc_the_khong_dau\",\n"
        "  \"knowledge_cluster\": \"...\",\n"
        "  \"metadata\": {\n"
        "    \"related_partners\": [\"...\"],\n"
        "    \"evidence\": {\"duration_months\": [], \"tuition_vnd\": []},\n"
        "    \"is_commercial\": true/false\n"
        "  }\n"
        "}\n"
        "\n"
        f"Tenant: {tenant_id}\n"
        f"Source file: {source_file}\n"
        "\n"
        "Danh sách thực thể gợi ý (có thể bỏ nếu không tìm thấy thông tin):\n"
        f"{entities_list}\n"
        "\n"
        "DỮ LIỆU THEO THỰC THỂ (đã gom từ toàn bộ tài liệu):\n"
        f"{prompt_body}\n"
        "\n"
        "DỮ LIỆU CHUNG (nếu có):\n"
        f"{general_block}\n"
    )


def build_entity_prompt(
    *,
    tenant_id: str,
    source_file: str,
    entity_id: str,
    entity_name: str,
    entity_context: str,
    general_context: str,
) -> str:
    """
    Smaller, per-entity prompt to reduce truncation and JSON syntax errors.
    Output MUST be a single JSON object.
    """
    ctx = (entity_context or "").strip()
    if len(ctx) > LLM_MAX_CHARS_PER_ENTITY:
        ctx = ctx[:LLM_MAX_CHARS_PER_ENTITY].rstrip() + "\n...[TRUNCATED]..."

    gctx = (general_context or "").strip()
    if gctx:
        gctx = gctx[:LLM_MAX_CHARS_PER_ENTITY].rstrip()

    return (
        "Bạn là một Document Architect chuyên nghiệp. Nhiệm vụ: tạo Integrated Knowledge Maps cho hệ thống RAG SaaS.\n"
        "Ràng buộc:\n"
        "- CHỈ dùng thông tin trong văn bản được cung cấp.\n"
        "- Không bịa đặt.\n"
        "- Output CHỈ là JSON OBJECT hợp lệ (không markdown, không giải thích).\n"
        "\n"
        "Yêu cầu cho kiến thức (knowledge_cluster):\n"
        "- Hãy viết như một chuyên gia tư vấn đang trả lời khách hàng.\n"
        "- Kết nối các mốc thời gian, tiền bạc và cam kết thực tế vào câu văn.\n"
        "- Ví dụ: \"Khóa học IELTS 6.5 học trong 4 tháng với học phí 10tr, cam kết học lại nếu trượt...\"\n"
        "\n"
        "Yêu cầu cho metadata.evidence (dictionary):\n"
        "- evidence là JSON OBJECT (không phải list).\n"
        "- Chỉ dùng các giá trị trích xuất được từ tài liệu.\n"
        "- Schema gợi ý:\n"
        "  {\"duration_months\": [number], \"duration_weeks\": [number], \"tuition_vnd\": [number], \"ielts_target\": [number], \"toeic_target\": [number], \"cefr_target\": [string]}\n"
        "- Không copy nguyên bảng học phí; nếu cần thì trích 1-3 dòng liên quan.\n"
        "\n"
        "Output format (JSON OBJECT):\n"
        "{\n"
        "  \"entity_id\": \"ten_thuc_the_khong_dau\",\n"
        "  \"knowledge_cluster\": \"...\",\n"
        "  \"metadata\": {\n"
        "    \"related_partners\": [\"...\"],\n"
        "    \"evidence\": {\"duration_months\": [], \"tuition_vnd\": []},\n"
        "    \"is_commercial\": true/false\n"
        "  }\n"
        "}\n"
        "\n"
        f"Tenant: {tenant_id}\n"
        f"Source file: {source_file}\n"
        f"Entity (must keep the same entity_id): {entity_id} ({entity_name})\n"
        "\n"
        "DỮ LIỆU THEO THỰC THỂ (trích toàn bộ tài liệu):\n"
        f"{ctx}\n"
        "\n"
        "DỮ LIỆU CHUNG (nếu cần để bổ sung ngữ cảnh):\n"
        f"{gctx}\n"
    )


# -------------------------
# Main
# -------------------------


def _heuristic_only_maps(
    tenant_id: str,
    source_file: str,
    entities: List[Tuple[str, str]],
    sections: List[Section],
) -> List[Dict[str, object]]:
    """
    Fallback without LLM: build detailed clusters by concatenating relevant contexts.
    """
    out: List[Dict[str, object]] = []

    used_sections: set[Section] = set()
    entity_items = [(eid, name) for (eid, name) in entities if eid != "thong_tin_chung_ve_to_chuc"]

    for eid, name in entity_items:
        picked = collect_entity_sections(sections, name)
        if not picked:
            continue
        used_sections.update(picked)

        ctx = collect_entity_context(sections, name)
        partners = _extract_partners(ctx)
        evidence = _extract_evidence_dict(ctx)
        is_commercial = _is_commercial_text(ctx)
        out.append(
            {
                "entity_id": eid,
                "knowledge_cluster": _render_heuristic_cluster(name, ctx),
                "metadata": {"related_partners": partners, "evidence": evidence, "is_commercial": is_commercial},
            }
        )

    # General/org-level info: sections that didn't map to any entity (or are clearly generic).
    general_sections = [
        s for s in sections if s not in used_sections and s.body.strip() and _section_kind(s.heading) == "general"
    ]
    general_ctx = _render_sections(general_sections)
    if general_ctx.strip():
        out.append(
            {
                "entity_id": "thong_tin_chung_ve_to_chuc",
                "knowledge_cluster": _render_heuristic_cluster("Thông tin chung về tổ chức", general_ctx),
                "metadata": {
                    "related_partners": _extract_partners(general_ctx),
                    "evidence": _extract_evidence_dict(general_ctx),
                    "is_commercial": False,
                },
            }
        )

    # Hard fallback: if we still produced nothing, dump the first N sections as general.
    if not out:
        fallback_ctx = collect_general_context(sections)
        out.append(
            {
                "entity_id": "thong_tin_chung_ve_to_chuc",
                "knowledge_cluster": _render_heuristic_cluster("Thông tin chung về tổ chức", fallback_ctx),
                "metadata": {
                    "related_partners": _extract_partners(fallback_ctx),
                    "evidence": _extract_evidence_dict(fallback_ctx),
                    "is_commercial": False,
                },
            }
        )

    return out


def _extract_partners(text: str) -> List[str]:
    """
    Heuristic partner extraction from text.
    """
    t = text or ""
    patterns = [
        ("Cambridge", r"(?i)\bcambridge\b"),
        ("Oxford", r"(?i)\boxford\b"),
        ("Macmillan", r"(?i)\bmacmillan\b"),
        ("British Council", r"(?i)\bbritish\s+council\b"),
        ("IDP", r"(?i)\bidp\b"),
        ("ETS", r"(?i)\bets\b"),
        ("Pearson", r"(?i)\bpearson\b"),
    ]
    found = []
    for label, pat in patterns:
        if re.search(pat, t):
            found.append(label)
    return found


def _parse_money_to_vnd(s: str) -> Optional[int]:
    """
    Parse common VN money notations to integer VND.
    Supports: 9.500.000, 9,500,000, 10tr, 10 triệu, 10.5tr.
    """
    if not s:
        return None
    raw = (s or "").strip().lower()
    raw = raw.replace("₫", "đ")
    raw = raw.replace("vnđ", "vnd")
    raw = raw.replace("đồng", "dong")

    # 10tr / 10.5tr
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*tr\b", raw)
    if m:
        v = float(m.group(1).replace(",", "."))
        return int(round(v * 1_000_000))

    # 10 triệu
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*tri[eệ]u", raw)
    if m:
        v = float(m.group(1).replace(",", "."))
        return int(round(v * 1_000_000))

    # 9.500.000 vnd / 9,500,000 vnd / 9500000
    m = re.search(r"(\d[\d\.\, ]{4,})\s*(đ|vnd|dong)\b", raw)
    if m:
        num = re.sub(r"[^\d]", "", m.group(1))
        if len(num) >= 5:
            return int(num)
    return None


def _extract_evidence_dict(text: str) -> Dict[str, object]:
    """
    Extract evidence as a dictionary for SaaS filtering (duration/tuition/targets).
    Only includes keys that have values.
    """
    t = text or ""
    evidence: Dict[str, List[object]] = {
        "duration_months": [],
        "duration_weeks": [],
        "tuition_vnd": [],
        "ielts_target": [],
        "toeic_target": [],
        "cefr_target": [],
    }

    # Duration
    for m in re.finditer(r"(?i)\b(\d+(?:[.,]\d+)?)\s*(thang|tháng)\b", t):
        try:
            evidence["duration_months"].append(float(m.group(1).replace(",", ".")))
        except Exception:
            pass
    for m in re.finditer(r"(?i)\b(\d+(?:[.,]\d+)?)\s*(tuan|tuần)\b", t):
        try:
            evidence["duration_weeks"].append(float(m.group(1).replace(",", ".")))
        except Exception:
            pass

    # Tuition patterns
    for m in re.finditer(r"(?i)(?:^|\s)(\d[\d\., ]{4,}\s*(?:₫|đ|vnd|dong)\b)", t):
        v = _parse_money_to_vnd(m.group(1))
        if v is not None:
            evidence["tuition_vnd"].append(int(v))
    for m in re.finditer(r"(?i)\b(\d+(?:[.,]\d+)?)\s*(tr|tri[eệ]u)\b", t):
        v = _parse_money_to_vnd(m.group(0))
        if v is not None:
            evidence["tuition_vnd"].append(int(v))

    # Targets
    for m in re.finditer(r"(?i)\bielts\s*(\d(?:[.,]\d)?)\b", t):
        try:
            evidence["ielts_target"].append(float(m.group(1).replace(",", ".")))
        except Exception:
            pass
    for m in re.finditer(r"(?i)\btoeic\s*(\d{3,4})\b", t):
        try:
            evidence["toeic_target"].append(int(m.group(1)))
        except Exception:
            pass
    for m in re.finditer(r"\b(A1|A2|B1|B2|C1|C2)\b", t):
        evidence["cefr_target"].append(m.group(1))

    # Normalize lists
    def _uniq_sorted(vals: List[object]) -> List[object]:
        uniq: List[object] = []
        seen = set()
        for x in vals:
            k = str(x)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(x)
        try:
            return sorted(uniq)  # type: ignore[arg-type]
        except Exception:
            return uniq

    for k in list(evidence.keys()):
        evidence[k] = _uniq_sorted(evidence[k])

    return {k: v for k, v in evidence.items() if v}


def _coerce_evidence(value: object, *, text_fallback: str) -> Dict[str, object]:
    """
    Normalize evidence to a dictionary.
    - dict: keep and enrich missing keys from text_fallback
    - list/str/other: parse from text
    """
    parsed = _extract_evidence_dict(text_fallback or "")
    if isinstance(value, dict):
        out = dict(value)
        for k, v in parsed.items():
            out.setdefault(k, v)
        return out
    if isinstance(value, list):
        joined = " | ".join([str(x) for x in value])
        return _extract_evidence_dict(joined + "\n" + (text_fallback or ""))
    if isinstance(value, str):
        return _extract_evidence_dict(value + "\n" + (text_fallback or ""))
    return parsed


def _is_commercial_text(text: str) -> bool:
    if "₫" in (text or ""):
        return True
    tn = _norm(text)
    if any(k in tn for k in ["hoc phi", "vnd", "dong", "trieu"]):
        return True
    if re.search(r"\d[\d\.\, ]{3,}\s*(vnd|dong)", tn):
        return True
    return False


def _render_heuristic_cluster(entity_name: str, ctx: str) -> str:
    """
    Render a professional, structured cluster without LLM.
    """
    ctx = (ctx or "").strip()
    lines = []
    lines.append(f"Thực thể: {entity_name}")
    partners = _extract_partners(ctx)
    evidence = _extract_evidence_dict(ctx)
    if partners:
        lines.append(f"Đối tác liên quan: {', '.join(partners)}")
    if evidence:
        pieces: List[str] = []
        if isinstance(evidence.get("duration_months"), list) and evidence["duration_months"]:
            pieces.append("thời lượng " + ", ".join([str(x) for x in evidence["duration_months"]]) + " tháng")
        if isinstance(evidence.get("tuition_vnd"), list) and evidence["tuition_vnd"]:
            # Show at most 3 amounts to avoid duplicating full tables.
            vals = [int(x) for x in evidence["tuition_vnd"] if isinstance(x, (int, float))][:3]
            pieces.append("học phí " + ", ".join([f"{v:,}".replace(",", ".") for v in vals]) + " VND")
        if isinstance(evidence.get("ielts_target"), list) and evidence["ielts_target"]:
            pieces.append("mục tiêu IELTS " + ", ".join([str(x) for x in evidence["ielts_target"]]))
        if isinstance(evidence.get("toeic_target"), list) and evidence["toeic_target"]:
            pieces.append("mục tiêu TOEIC " + ", ".join([str(x) for x in evidence["toeic_target"]]))
        if pieces:
            lines.append("Tóm tắt nhanh: " + "; ".join(pieces) + ".")
    lines.append("")
    lines.append("Thông tin liên quan (trích từ tài liệu):")
    lines.append(ctx if ctx else "(Không tìm thấy đoạn liên quan rõ ràng trong tài liệu.)")
    return "\n".join(lines).strip()


def _split_for_embedding(text: str) -> List[str]:
    """
    Split large clusters into smaller embedding nodes while preserving paragraph boundaries.
    """
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= int(EMBED_NODE_MAX_CHARS):
        return [t]

    paras = [p.strip() for p in re.split(r"\n{2,}", t) if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        chunks.append("\n\n".join(buf).strip())
        buf = []
        buf_len = 0

    max_chars = int(EMBED_NODE_MAX_CHARS)
    for p in paras:
        if len(p) > max_chars:
            flush()
            # Best-effort sentence splitting; hard-cut as last resort.
            sents = re.split(r"(?<=[\.\!\?])\s+", p)
            cur = ""
            for s in sents:
                s = s.strip()
                if not s:
                    continue
                if len(cur) + len(s) + 1 <= max_chars:
                    cur = (cur + " " + s).strip()
                else:
                    if cur:
                        chunks.append(cur)
                    cur = s
            if cur:
                while len(cur) > max_chars:
                    chunks.append(cur[:max_chars].rstrip())
                    cur = cur[max_chars:].lstrip()
                if cur:
                    chunks.append(cur)
            continue

        if buf and (buf_len + len(p) + 2 > max_chars):
            flush()
        buf.append(p)
        buf_len += len(p) + 2

    flush()

    min_chars = int(EMBED_NODE_MIN_CHARS)
    if len(chunks) >= 2 and len(chunks[-1]) < min_chars:
        chunks[-2] = (chunks[-2].rstrip() + "\n\n" + chunks[-1].lstrip()).strip()
        chunks.pop()

    return chunks


def main() -> None:
    _ensure_agent_env()

    in_dir = Path(INPUT_DIR)
    if not in_dir.exists():
        raise FileNotFoundError(f"Missing INPUT_DIR: {in_dir}")
    md_files = sorted([p for p in in_dir.glob("*.md") if p.is_file()])
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in: {in_dir}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_out = Path(OUTPUT_NODES_JSONL)
    nodes_out.parent.mkdir(parents=True, exist_ok=True)

    all_nodes: List[str] = []
    errors: List[Dict[str, object]] = []
    start_ts = time.time()
    if PROGRESS_LOG:
        print(
            f"Start: docs={len(md_files)} use_llm={USE_LLM} strategy={LLM_STRATEGY} "
            f"json_mode={LLM_REQUEST_JSON_MODE} max_tokens={LLM_MAX_TOKENS}",
            flush=True,
        )
    llm_stats = {
        "llm_enabled": bool(USE_LLM),
        "llm_strategy": str(LLM_STRATEGY),
        "llm_request_json_mode": bool(LLM_REQUEST_JSON_MODE),
        "docs_total": 0,
        "docs_with_llm_attempt": 0,
        "llm_calls": 0,
        "llm_success": 0,
        "llm_fallback_entities": 0,
    }

    for doc_idx, md_path in enumerate(md_files, start=1):
        llm_stats["docs_total"] += 1
        tenant_id = _derive_tenant_from_filename(md_path)
        source_file = md_path.name
        try:
            if PROGRESS_LOG:
                print(f"[{doc_idx}/{len(md_files)}] reading {source_file} (tenant={tenant_id})", flush=True)
            md_text = md_path.read_text(encoding="utf-8", errors="replace")
            sections = parse_markdown_sections(md_text)
            entities = discover_entities(sections)

            if not entities:
                entities = [("thong_tin_chung_ve_to_chuc", "Thông tin chung về tổ chức")]

            entity_contexts = []
            for eid, name in entities:
                if eid == "thong_tin_chung_ve_to_chuc":
                    continue
                ctx = collect_entity_context(sections, name)
                entity_contexts.append((eid, name, ctx))
            entity_contexts = _apply_budget_per_entity(entity_contexts)
            general_context = collect_general_context(sections)

            maps: List[Dict[str, object]] = []
            if USE_LLM and entity_contexts and str(LLM_STRATEGY).lower() == "per_entity":
                llm_stats["docs_with_llm_attempt"] += 1
                # For each entity, call LLM with a small prompt to reduce JSON failures.
                used_sections: set[Section] = set()
                for ent_idx, (eid, name, ctx) in enumerate(entity_contexts, start=1):
                    used_sections.update(collect_entity_sections(sections, name))
                    try:
                        if PROGRESS_LOG and PROGRESS_LOG_EVERY_ENTITY:
                            print(
                                f"  - [{doc_idx}/{len(md_files)}] entity {ent_idx}/{len(entity_contexts)}: {eid} calling LLM...",
                                flush=True,
                            )
                        prompt = build_entity_prompt(
                            tenant_id=tenant_id,
                            source_file=source_file,
                            entity_id=eid,
                            entity_name=name,
                            entity_context=ctx,
                            general_context=general_context,
                        )
                        llm_stats["llm_calls"] += 1
                        llm_text = _call_llm(prompt)
                        data = _extract_json(llm_text)
                        if isinstance(data, list):
                            data = next((x for x in data if isinstance(x, dict)), None)
                        if not isinstance(data, dict):
                            raise ValueError("LLM output is not a JSON object")
                        # Force entity_id consistency.
                        data["entity_id"] = eid
                        maps.append(data)
                        llm_stats["llm_success"] += 1
                        if PROGRESS_LOG and PROGRESS_LOG_EVERY_ENTITY:
                            print(f"    ok: {eid}", flush=True)
                    except Exception as e:
                        # Per-entity fallback to heuristic cluster.
                        llm_stats["llm_fallback_entities"] += 1
                        errors.append({"file": source_file, "tenant_id": tenant_id, "entity_id": eid, "error": f"LLM_entity_fallback: {e}"})
                        if PROGRESS_LOG and PROGRESS_LOG_EVERY_ENTITY:
                            print(f"    fallback: {eid} ({type(e).__name__})", flush=True)

                # Always add general info (heuristic) to satisfy the constraint, but avoid duplicating
                # content already captured by entity contexts.
                general_sections = [
                    s for s in sections if s not in used_sections and s.body.strip() and _section_kind(s.heading) == "general"
                ]
                general_ctx = _render_sections(general_sections)
                if general_ctx.strip():
                    maps.append(
                        {
                            "entity_id": "thong_tin_chung_ve_to_chuc",
                                "knowledge_cluster": _render_heuristic_cluster("Thông tin chung về tổ chức", general_ctx),
                                "metadata": {
                                    "related_partners": _extract_partners(general_ctx),
                                    "evidence": _extract_evidence_dict(general_ctx),
                                    "is_commercial": False,
                                },
                            }
                        )

            elif USE_LLM and entity_contexts and str(LLM_STRATEGY).lower() == "single_call":
                llm_stats["docs_with_llm_attempt"] += 1
                try:
                    prompt = build_prompt(
                        tenant_id=tenant_id,
                        source_file=source_file,
                        entity_contexts=entity_contexts,
                        general_context=general_context,
                    )
                    llm_stats["llm_calls"] += 1
                    llm_text = _call_llm(prompt)
                    data = _extract_json(llm_text)
                    if not isinstance(data, list):
                        raise ValueError("LLM output is not a JSON array")
                    maps = [x for x in data if isinstance(x, dict)]
                    llm_stats["llm_success"] += 1
                except Exception as e:
                    errors.append({"file": source_file, "tenant_id": tenant_id, "error": f"LLM_doc_fallback: {e}"})
                    maps = []

            # Fallback: heuristic-only if LLM disabled, produced empty output, or failed.
            if not maps:
                maps = _heuristic_only_maps(tenant_id, source_file, entities, sections)

            # Normalize + write per-doc JSON
            normalized = []
            for obj in maps:
                eid = obj.get("entity_id")
                if not isinstance(eid, str) or not eid.strip():
                    continue
                eid = _slugify(eid)
                if not eid:
                    continue
                cluster = obj.get("knowledge_cluster")
                if not isinstance(cluster, str):
                    cluster = str(cluster or "")
                cluster = cluster.strip()
                if not cluster:
                    continue
                md_meta = obj.get("metadata")
                if not isinstance(md_meta, dict):
                    md_meta = {}
                partners = md_meta.get("related_partners")
                evidence = md_meta.get("evidence")
                is_commercial = md_meta.get("is_commercial")
                if not isinstance(partners, list):
                    partners = []
                evidence_dict = _coerce_evidence(evidence, text_fallback=cluster)
                if not isinstance(evidence_dict, dict):
                    evidence_dict = {}
                # Sanitize values for JSONL stability
                sanitized_evidence: Dict[str, object] = {}
                for k, v in evidence_dict.items():
                    if isinstance(v, list):
                        sanitized_evidence[str(k)] = [
                            x if isinstance(x, (str, int, float, bool)) else str(x) for x in v if x is not None
                        ]
                    elif isinstance(v, (str, int, float, bool)) or v is None:
                        sanitized_evidence[str(k)] = v
                    else:
                        sanitized_evidence[str(k)] = str(v)
                if not isinstance(is_commercial, bool):
                    # Default: treat non-general entities as commercial
                    is_commercial = eid != "thong_tin_chung_ve_to_chuc"

                normalized.append(
                    {
                        "entity_id": eid,
                        "knowledge_cluster": cluster,
                        "metadata": {
                            "related_partners": [str(x) for x in partners if isinstance(x, (str, int, float))],
                            "evidence": sanitized_evidence,
                            "is_commercial": bool(is_commercial),
                        },
                    }
                )

                # Embedding-ready node lines (split long clusters to keep embeddings stable).
                chunks = _split_for_embedding(cluster)
                if not chunks:
                    continue
                chunk_count = len(chunks)
                base_id = f"{tenant_id}:{md_path.stem}:{eid}"
                for idx, chunk in enumerate(chunks, start=1):
                    node_id = f"{base_id}:{idx}" if chunk_count > 1 else base_id
                    node = {
                        "id": node_id,
                        "text": chunk,
                        "metadata": {
                            "tenant_id": tenant_id,
                            "source": source_file,
                            "source_stem": md_path.stem,
                            "entity_id": eid,
                            "related_partners": normalized[-1]["metadata"]["related_partners"],
                            "evidence": normalized[-1]["metadata"]["evidence"],
                            "is_commercial": normalized[-1]["metadata"]["is_commercial"],
                            "doc_type": "integrated_knowledge_map",
                            "chunk_index": idx,
                            "chunk_count": chunk_count,
                        },
                    }
                    all_nodes.append(json.dumps(node, ensure_ascii=False))

            per_doc_out = out_dir / f"{md_path.stem}.json"
            per_doc_out.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            errors.append({"file": source_file, "tenant_id": tenant_id, "error": str(e)})

    nodes_out.write_text("\n".join(all_nodes) + ("\n" if all_nodes else ""), encoding="utf-8")
    errors_path = out_dir / "_errors.json"
    if errors:
        errors_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    elif errors_path.exists():
        errors_path.unlink()

    stats_path = out_dir / "_run_stats.json"
    stats_path.write_text(json.dumps(llm_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed_s = time.time() - start_ts
    print(f"OK: wrote per-doc JSON to {out_dir}")
    print(f"OK: wrote embedding nodes JSONL to {nodes_out}")
    print(f"Done in {elapsed_s:.1f}s")
    print(
        f"LLM: enabled={llm_stats['llm_enabled']} strategy={llm_stats['llm_strategy']} "
        f"calls={llm_stats['llm_calls']} success={llm_stats['llm_success']} "
        f"entity_fallback={llm_stats['llm_fallback_entities']}"
    )
    if errors:
        print(f"WARN: some files failed; see {errors_path}")


if __name__ == "__main__":
    main()
