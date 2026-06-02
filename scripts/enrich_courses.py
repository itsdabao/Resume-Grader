import json
import os
import re
import random
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =========================
# CONFIG (edit if needed)
# =========================
INPUT_DIR = "data/knowledge_base"  # only *.pdf directly inside this folder
OUTPUT_JSONL = "data/.cache/enriched_courses.jsonl"
ERRORS_JSONL = "data/.cache/enriched_courses_errors.jsonl"

# LLM (uses your existing `.env` Groq/Gemini/OpenAI config)
LLM_MAX_CHARS_PER_SECTION = 8000
LLM_MAX_TOTAL_CHARS = 24000
LLM_RETRY_MAX = 5
LLM_RETRY_INITIAL_DELAY_S = 2.0
LLM_RETRY_BACKOFF = 2.0

# Heuristic fallback: if the PDF doesn't contain explicit "Mục 5/6/8/10", try to classify sections by keywords.
USE_HEURISTIC_FALLBACK = True
HEURISTIC_TOP_SECTIONS_PER_BUCKET = 4


def _derive_tenant_from_filename(path: Path) -> str:
    name = path.stem
    if name.lower().startswith("tenant_"):
        name = name[len("tenant_") :]
    safe = re.sub(r"[^a-z0-9_]+", "", name.lower())
    return safe or name.lower()


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
            if (block.startswith("{") and block.endswith("}")) or (block.startswith("[") and block.endswith("]")):
                s = block
                break
    # Best-effort: decode the first JSON value and ignore any trailing junk.
    first_candidates = [i for i in (s.find("["), s.find("{")) if i != -1]
    if not first_candidates:
        raise ValueError("No JSON object/array found in LLM output.")
    start = min(first_candidates)
    s2 = s[start:].lstrip()
    try:
        decoder = json.JSONDecoder()
        obj, _end = decoder.raw_decode(s2)
        return obj
    except Exception:
        # Fallback: slice between the outermost braces/brackets (may still fail if multiple JSON blocks exist).
        start_obj, end_obj = s.find("{"), s.rfind("}")
        start_arr, end_arr = s.find("["), s.rfind("]")
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            s = s[start_arr : end_arr + 1]
        elif start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            s = s[start_obj : end_obj + 1]
    return json.loads(s)

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _norm(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = _strip_accents(s).lower()
    return s


def _looks_like_markdown(text: str) -> bool:
    t = text or ""
    return ("\n# " in t) or ("\n## " in t) or t.lstrip().startswith("# ")


def _split_markdown_sections(markdown: str, *, heading_level: int = 2) -> List[Tuple[str, str]]:
    """
    Split markdown into sections by a chosen heading level. Returns (heading, body).
    """
    md = (markdown or "").replace("\r\n", "\n").replace("\r", "\n")
    if not md.strip():
        return []
    level = max(1, min(6, int(heading_level)))
    pat = re.compile(rf"^(#{{{level}}})\s+(.+?)\s*$")
    lines = md.splitlines()

    out: List[Tuple[str, List[str]]] = []
    cur_h = ""
    cur: List[str] = []

    def flush():
        nonlocal cur, cur_h
        body = "\n".join(cur).strip()
        if body:
            out.append((cur_h.strip() or "Thông tin chung", cur.copy()))
        cur = []

    for ln in lines:
        m = pat.match(ln)
        if m:
            flush()
            cur_h = m.group(2).strip()
            cur = [ln]
        else:
            cur.append(ln)
    flush()
    return [(h, "\n".join(ls).strip()) for h, ls in out]


def _split_plaintext_sections(text: str) -> List[Tuple[str, str]]:
    """
    Heuristic splitter for plain text PDFs without markdown headings.
    Returns (heading, body).
    """
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not s.strip():
        return []
    lines = [ln.rstrip() for ln in s.splitlines()]

    re_numbered = re.compile(r"^\s*(\d+(\.\d+){0,3}|\d+)\s*[\)\.\-:]\s+\S")
    re_roman = re.compile(r"^\s*[IVXLCDM]{1,8}\s*[\)\.\-:]\s+\S", re.IGNORECASE)
    re_label = re.compile(r"^\s*(PHẦN|CHƯƠNG|MỤC|CHUYÊN ĐỀ|GIỚI THIỆU)\b", re.IGNORECASE)
    re_semantic_heading = re.compile(
        r"^\s*(hoc\s*phi|phi\s*tai\s*lieu|lich(\s*khai\s*giang|\s*hoc)?|khai\s*giang|chinh\s*sach|quy\s*dinh|cam\s*ket|khoa\s*hoc|chuong\s*trinh|lo\s*trinh)\b",
        re.IGNORECASE,
    )

    def is_heading(line: str) -> bool:
        ln = (line or "").strip()
        if not ln or len(ln) > 90:
            return False
        # Semantic headings (accent-insensitive) like "Học phí", "Lịch khai giảng", ...
        if re_semantic_heading.match(_strip_accents(ln).lower()):
            return True
        if re_numbered.match(ln) or re_roman.match(ln) or re_label.match(ln):
            return True
        letters = [ch for ch in ln if ch.isalpha()]
        if letters and len(letters) >= 6:
            upper_ratio = sum(1 for ch in letters if ch == ch.upper()) / float(len(letters))
            if upper_ratio >= 0.85 and len(ln.split()) <= 12:
                return True
        if ln.endswith(":") and len(ln.split()) <= 10:
            return True
        return False

    sections: List[Tuple[str, List[str]]] = []
    cur_h = ""
    cur: List[str] = []

    def flush():
        nonlocal cur, cur_h
        body = "\n".join(cur).strip()
        if body:
            sections.append((cur_h.strip() or "Thông tin chung", cur.copy()))
        cur = []

    for ln in lines:
        if is_heading(ln):
            flush()
            cur_h = ln.strip()
            cur = [ln]
        else:
            cur.append(ln)
    flush()

    # For enrichment classification we do NOT merge short sections; small headings like "HỌC PHÍ"
    # are often extremely informative and must be kept separate.
    return [(h, "\n".join(ls).strip()) for h, ls in sections if ls]


def _split_sections(text: str) -> List[Tuple[str, str]]:
    if _looks_like_markdown(text):
        secs = _split_markdown_sections(text, heading_level=2)
        return secs if secs else [("Document", text)]
    secs = _split_plaintext_sections(text)
    return secs if secs else [("Document", text)]


_KW_COURSE = [
    "khoa hoc",
    "lo trinh",
    "muc tieu",
    "dau vao",
    "dau ra",
    "cap do",
    "chuong trinh",
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
]
_KW_TUITION = [
    "hoc phi",
    "phi",
    "le phi",
    "phi tai lieu",
    "tai lieu",
    "uu dai",
    "giam",
    "khuyen mai",
    "dong",
    "vnd",
    "trieu",
]
_KW_POLICY = [
    "chinh sach",
    "quy dinh",
    "cam ket",
    "noi quy",
    "bao luu",
    "hoan phi",
    "hoan tien",
    "doi lop",
    "chuyen lop",
    "hoc vien",
]
_KW_SCHEDULE = [
    "lich",
    "khai giang",
    "lich hoc",
    "thoi khoa bieu",
    "thu",
    "ca hoc",
    "gio hoc",
    "buoi",
    "tuan",
    "ngay",
]


def _keyword_score(text_norm: str, keywords: List[str]) -> int:
    score = 0
    for kw in keywords:
        if not kw:
            continue
        if kw in text_norm:
            # reward multiple occurrences lightly
            score += 2 + min(8, text_norm.count(kw))
    return score


def _score_section(heading: str, body: str) -> Dict[str, float]:
    hn = _norm(heading)
    bn = _norm(body)
    txt = hn + "\n" + bn

    # Pattern boosts
    has_money = bool(re.search(r"(?i)\b(vnd|đ|dong|triệu|trieu)\b", txt)) or bool(re.search(r"\d[\d\.\, ]{3,}\s*(đ|vnd|dong)", txt))
    has_date = bool(re.search(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b", txt)) or ("khai giang" in txt)
    has_time = bool(re.search(r"\b\d{1,2}[:h]\d{0,2}\b", txt))

    s_course = _keyword_score(txt, _KW_COURSE)
    s_tuition = _keyword_score(txt, _KW_TUITION) + (6 if has_money else 0)
    s_policy = _keyword_score(txt, _KW_POLICY)
    s_schedule = _keyword_score(txt, _KW_SCHEDULE) + (5 if has_date else 0) + (2 if has_time else 0)

    # Heading boosts (more important than body)
    s_course += 2 * _keyword_score(hn, _KW_COURSE)
    s_tuition += 2 * _keyword_score(hn, _KW_TUITION)
    s_policy += 2 * _keyword_score(hn, _KW_POLICY)
    s_schedule += 2 * _keyword_score(hn, _KW_SCHEDULE)

    return {
        "course_info": float(s_course),
        "tuition": float(s_tuition),
        "policy": float(s_policy),
        "schedule": float(s_schedule),
    }


def _extract_sections_heuristic(text: str) -> Tuple[Dict[str, str], Dict[str, object]]:
    """
    Heuristically classify sections into 4 buckets and return joined texts + debug info.
    """
    secs = _split_sections(text)
    # Compute scores once per section, then rank with cross-penalties to reduce mis-bucketing.
    per_section: List[Tuple[str, str, Dict[str, float]]] = []
    for heading, body in secs:
        per_section.append((heading, body, _score_section(heading, body)))

    ranked: Dict[str, List[Tuple[float, float, str, str]]] = {
        "course_info": [],  # (adjusted, raw, heading, body)
        "tuition": [],
        "policy": [],
        "schedule": [],
    }
    for heading, body, scores in per_section:
        for k in ranked.keys():
            raw = float(scores.get(k, 0.0))
            other_max = max(float(v) for kk, v in scores.items() if kk != k)
            # Penalize if another bucket strongly dominates this section.
            adjusted = raw - 0.7 * other_max
            ranked[k].append((adjusted, raw, heading, body))

    selected: Dict[str, List[Tuple[float, str]]] = {}
    out: Dict[str, str] = {}
    for k, items in ranked.items():
        items.sort(key=lambda x: x[0], reverse=True)
        top = [it for it in items if it[0] > 0][:HEURISTIC_TOP_SECTIONS_PER_BUCKET]
        selected[k] = [(raw, hd) for _adj, raw, hd, _ in top]
        parts = []
        for adj, raw, hd, bd in top:
            parts.append(f"[{hd}] (score={raw:.0f}, adj={adj:.1f})\n{bd}")
        out[k] = "\n\n---\n\n".join(parts).strip()

    dbg = {
        "n_sections_total": len(secs),
        "selected_headings": selected,
    }
    return out, dbg


def _parse_tuition(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    s = value.strip().lower()
    if not s:
        return None

    # Handle "x triệu" shorthand
    if "triệu" in s:
        m = re.search(r"(\d+(?:[.,]\d+)?)", s)
        if m:
            try:
                num = float(m.group(1).replace(",", "."))
                return float(int(num * 1_000_000))
            except Exception:
                pass

    # Default: take digits only (treat separators/currency as formatting)
    digits = re.findall(r"\d+", s)
    if not digits:
        return None
    try:
        return float(int("".join(digits)))
    except Exception:
        return None


def _find_section(text: str, section_no: int) -> Optional[Tuple[int, int]]:
    """
    Find a slice for "Mục <n>" style sections in Vietnamese docs.
    Returns (start_idx, end_idx) in the original text, or None.
    """
    t = text or ""
    # Common patterns: "Mục 5", "MUC 5", "5. ", "5) "
    start_patterns = [
        rf"(?mi)^\s*m[uụ]c\s*0*{section_no}\b.*$",
        rf"(?mi)^\s*0*{section_no}\s*[\)\.\-:]\s+.*$",
    ]
    end_patterns = [
        rf"(?mi)^\s*m[uụ]c\s*0*{section_no + 1}\b.*$",
        rf"(?mi)^\s*0*{section_no + 1}\s*[\)\.\-:]\s+.*$",
    ]

    start_idx = None
    for pat in start_patterns:
        m = re.search(pat, t)
        if m:
            start_idx = m.start()
            break
    if start_idx is None:
        return None

    end_idx = len(t)
    for pat in end_patterns:
        m = re.search(pat, t[start_idx + 1 :])
        if m:
            end_idx = start_idx + 1 + m.start()
            break
    return start_idx, end_idx


def _extract_sections(text: str) -> Dict[str, str]:
    """
    Pull the 4 relevant sections if possible; otherwise fall back to full text.
    """
    out: Dict[str, str] = {}
    sections = {
        "course_info": 5,
        "tuition": 6,
        "policy": 8,
        "schedule": 10,
    }
    for key, no in sections.items():
        sl = _find_section(text, no)
        if sl is None:
            out[key] = ""
        else:
            out[key] = (text[sl[0] : sl[1]] or "").strip()
    # If nothing matched, provide full text to course_info as a fallback.
    if not any(v.strip() for v in out.values()):
        out["course_info"] = (text or "").strip()
    return out


def _clamp(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "\n...[TRUNCATED]..."


def _apply_prompt_budgets(sections: Dict[str, str]) -> Dict[str, str]:
    """
    Apply both per-section and total prompt budget.
    """
    keys = ["course_info", "tuition", "policy", "schedule"]
    clamped = {k: _clamp(sections.get(k, ""), LLM_MAX_CHARS_PER_SECTION) for k in keys}

    total = sum(len(clamped[k]) for k in keys)
    if total <= LLM_MAX_TOTAL_CHARS:
        return clamped

    # Reduce proportionally while keeping a small minimum for non-empty sections.
    min_each = 1200
    present = [k for k in keys if clamped[k].strip()]
    budgets: Dict[str, int] = {}
    remaining = int(LLM_MAX_TOTAL_CHARS)

    for k in present:
        budgets[k] = min(min_each, len(clamped[k]))
        remaining -= budgets[k]
    remaining = max(0, remaining)

    extras = {k: max(0, len(clamped[k]) - budgets.get(k, 0)) for k in present}
    sum_extras = sum(extras.values()) or 1
    for k in present:
        extra_alloc = int(remaining * (extras[k] / sum_extras))
        budgets[k] = min(len(clamped[k]), budgets.get(k, 0) + extra_alloc)

    out: Dict[str, str] = {}
    for k in keys:
        s = clamped[k]
        b = budgets.get(k, len(s))
        if len(s) > b:
            out[k] = s[:b].rstrip() + "\n...[TRUNCATED]..."
        else:
            out[k] = s
    return out


def _call_llm(prompt: str) -> str:
    """
    Use the same provider selection as the main app via `app.core.llama.init_llm_from_env`.
    This will call Groq/Gemini/OpenAI depending on `.env`.
    """
    from llama_index.core import Settings

    # Avoid `Settings.llm` property auto-resolving to OpenAI default when OPENAI_API_KEY is missing.
    llm = getattr(Settings, "_llm", None)
    if llm is None:
        from app.core.llama import init_llm_from_env

        init_llm_from_env()
        llm = getattr(Settings, "_llm", None)
    if llm is None:
        raise RuntimeError("LLM is not initialized. Check your `.env` LLM_PROVIDER + API keys.")

    delay = float(LLM_RETRY_INITIAL_DELAY_S)
    last_err: Exception | None = None
    for attempt in range(1, int(LLM_RETRY_MAX) + 1):
        try:
            resp = llm.complete(prompt)
            return getattr(resp, "text", None) or str(resp)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            retryable = any(
                k in msg for k in ["timeout", "timed out", "429", "rate", "quota", "temporarily", "connection"]
            )
            if attempt >= int(LLM_RETRY_MAX) or not retryable:
                break
            sleep_s = delay + random.uniform(0.0, 0.6)
            time.sleep(sleep_s)
            delay *= float(LLM_RETRY_BACKOFF)
    raise RuntimeError(f"LLM call failed after {LLM_RETRY_MAX} attempts: {last_err}") from last_err


def _build_prompt(
    *,
    tenant_id: str,
    source_pdf: str,
    course_info: str,
    tuition: str,
    policy: str,
    schedule: str,
) -> str:
    return (
        "Bạn là chuyên gia RAG. Bạn nhận 4 đoạn tài liệu rời rạc (cắt từ 1 PDF) và phải tổng hợp thành Enriched Chunks.\n"
        "Ràng buộc:\n"
        "- CHỈ dùng thông tin có trong các đoạn tài liệu được cung cấp.\n"
        "- KHÔNG bịa số liệu hoặc ngày tháng.\n"
        "- Nếu thiếu lịch khai giảng cụ thể, đặt next_opening = \"Liên hệ để cập nhật lịch mới nhất\".\n"
        "- Output CHỈ là JSON hợp lệ (không markdown, không giải thích).\n"
        "\n"
        "Nhiệm vụ: Với mỗi khóa học xuất hiện trong tài liệu, tạo 1 object theo format:\n"
        "{\n"
        "  \"course_id\": \"string\",\n"
        "  \"enriched_content\": \"string\",\n"
        "  \"metadata\": {\n"
        "    \"category\": \"string\",\n"
        "    \"tuition\": number|null,\n"
        "    \"next_opening\": \"string\"\n"
        "  }\n"
        "}\n"
        "\n"
        "Quy tắc cho course_id:\n"
        f"- Bắt buộc prefix bằng tenant_id \"{tenant_id}\".\n"
        "- Dạng gợi ý: <tenant_id>:<slug_khoa_hoc> (slug chỉ gồm a-z0-9_).\n"
        "\n"
        "Quy tắc cho metadata:\n"
        f"- category = \"{tenant_id}\".\n"
        "- tuition: tổng học phí (bao gồm phí tài liệu) nếu tài liệu có đủ số liệu; nếu không đủ thì null.\n"
        "- next_opening: lịch khai giảng gần nhất nếu có; nếu không có thì dùng câu mặc định ở trên.\n"
        "\n"
        "Yêu cầu enriched_content (1 đoạn tổng hợp duy nhất):\n"
        "1) Tên khóa học + đầu vào + đầu ra.\n"
        "2) Thời lượng học + tổng học phí (kèm ghi chú \"(bao gồm phí tài liệu)\" nếu có).\n"
        "3) Chính sách cam kết/quy định áp dụng cho khóa học (nếu có).\n"
        "4) Lịch khai giảng và lịch học cụ thể gần nhất.\n"
        "\n"
        f"PDF nguồn: {source_pdf}\n"
        "\n"
        "=== INPUT: Mục 5 - Thông tin khóa học ===\n"
        + course_info
        + "\n\n=== INPUT: Mục 6 - Học phí ===\n"
        + tuition
        + "\n\n=== INPUT: Mục 8 - Chính sách ===\n"
        + policy
        + "\n\n=== INPUT: Mục 10 - Lịch khai giảng ===\n"
        + schedule
        + "\n\nOUTPUT JSON:\n"
        "Trả về một JSON ARRAY, mỗi phần tử là 1 object theo format ở trên.\n"
    )


def _extract_pdf_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency `pypdf`. Recommended to run via:\n"
            "  conda run -n agent python scripts/enrich_courses.py\n"
            "Or install into your current env:\n"
            "  pip install pypdf"
        ) from e

    reader = PdfReader(str(pdf_path))
    parts: List[str] = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts).strip()


def _ensure_agent_env() -> None:
    """
    If user runs in an env without llama-index, re-exec with conda env `agent`.
    """
    if os.getenv("ENRICH_NO_REEXEC"):
        return
    try:
        import llama_index  # type: ignore  # noqa: F401
        return
    except Exception:
        pass
    cmd = ["conda", "run", "-n", "agent", "python", str(Path(__file__).resolve())]
    env = dict(os.environ)
    env["ENRICH_NO_REEXEC"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    print("Re-running with: " + " ".join(cmd))
    try:
        r = subprocess.run(cmd, env=env)
        raise SystemExit(r.returncode)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Cannot find `conda` to re-exec into env `agent`. "
            "Please run this script inside the correct environment:\n"
            "  conda run -n agent python scripts/enrich_courses.py"
        ) from e


def main() -> None:
    _ensure_agent_env()

    in_dir = Path(INPUT_DIR)
    if not in_dir.exists():
        raise FileNotFoundError(f"Missing INPUT_DIR: {in_dir}")

    pdfs = sorted([p for p in in_dir.glob("*.pdf") if p.is_file()])
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in: {in_dir}")

    out_path = Path(OUTPUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path = Path(ERRORS_JSONL)
    err_path.parent.mkdir(parents=True, exist_ok=True)

    out_lines: List[str] = []
    err_lines: List[str] = []
    for pdf in pdfs:
        tenant_id = _derive_tenant_from_filename(pdf)
        try:
            raw_text = _extract_pdf_text(pdf)
            secs = _extract_sections(raw_text)

            heuristic_dbg = None
            # If the PDF doesn't follow the "Mục 5/6/8/10" template, use heuristic bucketization.
            if USE_HEURISTIC_FALLBACK:
                missing = [k for k in ("course_info", "tuition", "policy", "schedule") if not (secs.get(k, "") or "").strip()]
                if len(missing) >= 2:
                    heur_secs, heuristic_dbg = _extract_sections_heuristic(raw_text)
                    for k in ("course_info", "tuition", "policy", "schedule"):
                        if not (secs.get(k, "") or "").strip():
                            secs[k] = heur_secs.get(k, "")

            budgets = _apply_prompt_budgets(secs)
            section_found = {k: bool((secs.get(k, "") or "").strip()) for k in ["course_info", "tuition", "policy", "schedule"]}

            prompt = _build_prompt(
                tenant_id=tenant_id,
                source_pdf=pdf.name,
                course_info=budgets.get("course_info", ""),
                tuition=budgets.get("tuition", ""),
                policy=budgets.get("policy", ""),
                schedule=budgets.get("schedule", ""),
            )

            llm_text = _call_llm(prompt)
            data = _extract_json(llm_text)
            if not isinstance(data, list):
                raise ValueError("LLM output is not a JSON array")

            for obj in data:
                if not isinstance(obj, dict):
                    continue
                course_id = obj.get("course_id")
                enriched_content = obj.get("enriched_content")
                metadata = obj.get("metadata")
                if not isinstance(course_id, str) or not course_id.strip():
                    continue
                if not course_id.lower().startswith(tenant_id.lower() + ":"):
                    slug = re.sub(r"[^a-z0-9_]+", "_", course_id.lower()).strip("_")
                    course_id = f"{tenant_id}:{slug}" if slug else f"{tenant_id}:{course_id.lower()}"
                if not isinstance(enriched_content, str):
                    enriched_content = str(enriched_content or "")
                enriched_content = enriched_content.strip()
                if not enriched_content:
                    continue
                if not isinstance(metadata, dict):
                    metadata = {}

                category = metadata.get("category")
                if not isinstance(category, str) or not category.strip():
                    category = tenant_id

                next_opening = metadata.get("next_opening")
                if not isinstance(next_opening, str) or not next_opening.strip():
                    next_opening = "Liên hệ để cập nhật lịch mới nhất"

                tuition_num = _parse_tuition(metadata.get("tuition"))

                out_obj = {
                    "course_id": course_id,
                    "enriched_content": enriched_content,
                    "metadata": {
                        "category": category,
                        "tuition": tuition_num,
                        "next_opening": next_opening,
                    },
                }
                out_lines.append(json.dumps(out_obj, ensure_ascii=False))
        except Exception as e:
            err_lines.append(
                json.dumps(
                    {
                        "file": pdf.name,
                        "tenant_id": tenant_id,
                        "error": str(e),
                        "section_found": section_found if "section_found" in locals() else None,
                        "heuristic_dbg": heuristic_dbg,
                    },
                    ensure_ascii=False,
                )
            )

    out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    err_path.write_text("\n".join(err_lines) + ("\n" if err_lines else ""), encoding="utf-8")
    print(f"OK: wrote {out_path}")
    if err_lines:
        print(f"WARN: some files failed; see {err_path}")


if __name__ == "__main__":
    main()
