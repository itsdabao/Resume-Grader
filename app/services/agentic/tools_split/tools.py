from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llama_index.core import VectorStoreIndex

from app.core.config import PROJECT_ROOT, RETRIEVAL_TOP_K
from app.services.rag.incontext_ralm import query_with_incontext_ralm, retrieve_hybrid_contexts

from .evidence import extract_evidence_dict, parse_money_to_vnd
from .fee_extractor import extract_financials_with_llm, refine_extracted_fees
from .preprocess import extract_phone
from . import policy_registry


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolResult:
    answer: str
    sources: List[str]
    metadata: Dict[str, object] | None = None
    context_texts: List[str] | None = None


def _sources_from_contexts(contexts: List[Dict[str, object]]) -> List[str]:
    sources: List[str] = []
    seen = set()
    for r in contexts:
        m = r.get("meta", {}) or {}
        src = m.get("file_name") or m.get("file_path") or m.get("source") or "unknown"
        s = str(src)
        if s in seen:
            continue
        seen.add(s)
        sources.append(s)
    return sources


def _merge_context_dicts(
    primary: List[Dict[str, object]],
    secondary: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen: set[str] = set()
    for row in (primary or []) + (secondary or []):
        if not isinstance(row, dict):
            continue
        text = str(row.get("text", "") or "")
        sig = _norm_ascii_text(text)[:240]
        if not sig:
            continue
        if sig in seen:
            continue
        seen.add(sig)
        out.append(row)
    return out


def _norm_ascii_text(s: str) -> str:
    t = unicodedata.normalize("NFD", s or "")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = t.replace("đ", "d").replace("Đ", "D").lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def _safe_parse_grouped_vnd(raw: str) -> Optional[int]:
    m = re.search(r"\b(\d{1,3}(?:\.\d{3})+)\b", str(raw or ""))
    if not m:
        return None
    try:
        v = int(m.group(1).replace(".", ""))
    except Exception:
        return None
    return v if v > 0 else None


def _extract_focus_course_from_query(question: str) -> Optional[str]:
    qn = _norm_ascii_text(question or "")
    m = re.search(r"\bkhoa\s+([a-z0-9][a-z0-9\s]{2,80})", qn)
    if not m:
        return None
    tail = m.group(1)
    cut_at = len(tail)
    stop_patterns = [
        r"\bhoc phi\b",
        r"\bgia\b",
        r"\bbao nhieu\b",
        r"\bneu\b",
        r"\bthi\b",
        r"\bva\b",
        r"\broi\b",
        r"\bco cong don\b",
        r"\bkhong\b",
        r"\bcua\b",
        r"\btheo\b",
    ]
    for pat in stop_patterns:
        ms = re.search(pat, tail)
        if ms:
            cut_at = min(cut_at, int(ms.start()))
    tail = tail[:cut_at]
    focus = re.sub(r"\s+", " ", tail).strip()
    if not focus or len(focus) < 3:
        return None
    return focus


def _course_tokens(course: str) -> List[str]:
    stop = {"khoa", "hoc", "phi", "lop", "gia", "bao", "nhieu", "trung", "tam", "course"}
    toks = [t for t in _norm_ascii_text(course or "").split() if len(t) >= 3 and t not in stop]
    return toks[:6]


def _text_mentions_course(*, course: str, text: str) -> bool:
    toks = _course_tokens(course)
    if not toks:
        return False
    txt = _norm_ascii_text(text or "")
    hits = sum(1 for t in toks if re.search(rf"\b{re.escape(t)}\b", txt))
    return hits >= min(2, len(toks))


def _extract_course_fee_pairs_from_text(text: str) -> Dict[str, int]:
    """
    Parse simple lines like:
    - Pronunciation Bootcamp: 5.500.000
    - Hoc phi Khoa IELTS Intermediate: 11.000.000 VND
    """
    out: Dict[str, int] = {}
    pending_name: Optional[str] = None
    for ln in (text or "").splitlines():
        line = ln.strip()
        if not line:
            pending_name = None
            continue
        if ":" in line:
            left, right = line.split(":", 1)
            clean_name = re.sub(r"(?i)\b(hoc phi|khoa|lop)\b", " ", left)
            clean_name = re.sub(r"\s+", " ", clean_name).strip(" -\t")
            pending_name = clean_name if len(clean_name) >= 3 else None
            v = parse_money_to_vnd(right.strip())
            if v is None:
                v = _safe_parse_grouped_vnd(right.strip())
            if v is None or int(v) < 1_000_000 or not pending_name:
                continue
            key = _norm_ascii_text(pending_name)
            if not key:
                continue
            cur = out.get(key)
            out[key] = int(v) if cur is None else max(int(cur), int(v))
            pending_name = None
            continue

        if pending_name:
            v = parse_money_to_vnd(line)
            if v is None:
                v = _safe_parse_grouped_vnd(line)
            if v is None or int(v) < 1_000_000:
                continue
            key = _norm_ascii_text(pending_name)
            if key:
                cur = out.get(key)
                out[key] = int(v) if cur is None else max(int(cur), int(v))
            pending_name = None
    return out


def _match_course_price(*, course_hint: str, course_price_map: Dict[str, int]) -> Optional[int]:
    toks = _course_tokens(course_hint)
    if not toks:
        return None
    best_key = None
    best_hits = 0
    for k in course_price_map.keys():
        hits = sum(1 for t in toks if re.search(rf"\b{re.escape(t)}\b", k))
        if hits > best_hits:
            best_hits = hits
            best_key = k
    if best_key is None or best_hits == 0:
        return None
    return int(course_price_map.get(best_key) or 0) or None


def _extract_fee_by_label_from_text(*, text: str, labels: List[str]) -> Optional[int]:
    labels_n = [_norm_ascii_text(x) for x in (labels or []) if str(x).strip()]
    if not labels_n:
        return None
    vals: List[int] = []
    for ln in (text or "").splitlines():
        nln = _norm_ascii_text(ln)
        if not any(lbl in nln for lbl in labels_n):
            continue
        v = parse_money_to_vnd(ln)
        if v is None:
            v = _safe_parse_grouped_vnd(ln)
        if v is None:
            continue
        if int(v) > 0:
            vv = int(v)
            # OCR noise guard: values like 300012 should map to 300000 for small fees.
            if vv < 2_000_000 and (vv % 1000) != 0:
                vv = int(round(vv / 1000.0) * 1000)
            vals.append(vv)
    if not vals:
        return None
    freq: Dict[int, int] = {}
    for v in vals:
        freq[int(v)] = int(freq.get(int(v), 0)) + 1
    best_count = max(freq.values())
    best_vals = [int(v) for v, c in freq.items() if int(c) == int(best_count)]
    return max(best_vals) if best_vals else max(vals)


def _extract_combo_discount_percent(text: str) -> Optional[float]:
    vals: List[float] = []
    lines = [str(ln or "") for ln in (text or "").splitlines()]
    for i, ln in enumerate(lines):
        ln_norm = _norm_ascii_text(ln)
        if "combo" not in ln_norm:
            continue
        own_vals = _extract_discount_percents_from_text(ln)
        for p in own_vals:
            if 0.0 < p <= 50.0:
                vals.append(float(p))
        if own_vals:
            continue
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            nxt_stripped = nxt.strip()
            nxt_norm = _norm_ascii_text(nxt)
            is_separate_bullet = nxt_stripped.startswith("-") or nxt_stripped.startswith("*") or bool(
                re.match(r"^\d+[.)]\s+", nxt_stripped)
            )
            if ("combo" in nxt_norm) or (not is_separate_bullet and nxt_stripped):
                for p in _extract_discount_percents_from_text(nxt):
                    if 0.0 < p <= 50.0:
                        vals.append(float(p))
    if vals:
        return max(vals)
    m = re.search(r"(?is)\bcombo\b[\s\S]{0,120}?\bgiam\s*(\d{1,2}(?:[.,]\d+)?)\s*%", text or "")
    if not m:
        return None
    try:
        p = float(m.group(1).replace(",", "."))
    except Exception:
        return None
    return p if 0.0 < p <= 50.0 else None


def _extract_discount_percents_from_text(text: str) -> List[float]:
    vals: List[float] = []
    for m in re.finditer(r"(?i)(\d{1,2}(?:[.,]\d+)?)\s*%", text or ""):
        try:
            vals.append(float(m.group(1).replace(",", ".")))
        except Exception:
            continue
    return sorted(set([x for x in vals if 0.0 < x <= 100.0]))


def _contains_injection_cues(question: str) -> bool:
    qn = _norm_ascii_text(question or "")
    cues = [
        "bo qua",
        "ignore",
        "khong can",
        "xac nhan luon",
        "tra loi gia",
        "khong theo tai lieu",
    ]
    return any(c in qn for c in cues)


def _sanitize_public_answer(text: str) -> str:
    raw = str(text or "")
    n = _norm_ascii_text(raw)
    if "lead data" not in n and "khi va chi khi" not in n:
        return raw.strip()
    lines = raw.splitlines()
    cleaned: List[str] = []
    skip = False
    for ln in lines:
        nn = _norm_ascii_text(ln)
        if "khi va chi khi" in nn:
            break
        if "lead data" in nn:
            skip = True
            continue
        if skip and ln.strip().startswith("```"):
            skip = False
            continue
        if skip:
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip()


def _extract_course_catalog_from_texts(texts: List[str]) -> Dict[str, Dict[str, object]]:
    """
    Return map keyed by normalized course name:
    {key: {"name": display_name, "duration_months": float|None, "tuition_vnd": int|None}}
    """
    catalog: Dict[str, Dict[str, object]] = {}
    for txt in texts or []:
        block = str(txt or "")
        plain = unicodedata.normalize("NFD", block)
        plain = "".join(ch for ch in plain if unicodedata.category(ch) != "Mn")
        plain = plain.replace("đ", "d").replace("Đ", "D")
        for m in re.finditer(r"(?i)\b\d+\.\d+\.\s*([^\(\n]{3,80})\((\d+(?:[.,]\d+)?)\s*(?:thang)\b", plain):
            name_n = re.sub(r"\s+", " ", (m.group(1) or "")).strip()
            if not name_n:
                continue
            key = _norm_ascii_text(name_n)
            try:
                dur = float((m.group(2) or "").replace(",", "."))
            except Exception:
                dur = None
            item = catalog.get(key) or {"name": name_n.title(), "duration_months": None, "tuition_vnd": None}
            if isinstance(dur, float):
                item["duration_months"] = dur
            catalog[key] = item
        fee_map = _extract_course_fee_pairs_from_text(block)
        for fee_key, fee_val in fee_map.items():
            item = catalog.get(fee_key) or {"name": fee_key.title(), "duration_months": None, "tuition_vnd": None}
            item["tuition_vnd"] = int(fee_val)
            catalog[fee_key] = item
    return catalog


def _fmt_vnd_public(v: int) -> str:
    return f"{int(v):,}".replace(",", ".") + " VND"


_NON_STACK_POLICY_CACHE: Dict[str, bool] = {}


def _tenant_policy_has_non_stack_rule(*, tenant_id: Optional[str]) -> bool:
    tid = str(tenant_id or "").strip().lower()
    if not tid:
        return False
    if tid in _NON_STACK_POLICY_CACHE:
        return bool(_NON_STACK_POLICY_CACHE[tid])
    nodes_path = Path(PROJECT_ROOT) / "data" / ".cache" / tid / "nodes.jsonl"
    if not nodes_path.exists():
        _NON_STACK_POLICY_CACHE[tid] = False
        return False
    hit = False
    try:
        with nodes_path.open("r", encoding="utf-8") as f:
            for ln in f:
                line = str(ln or "").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                txt = _norm_ascii_text(str(obj.get("text", "") or ""))
                if ("khong duoc cong don" in txt) or ("khong cong don" in txt):
                    hit = True
                    break
    except Exception:
        hit = False
    _NON_STACK_POLICY_CACHE[tid] = bool(hit)
    return bool(hit)


def course_search_tool(
    question: str,
    *,
    index: VectorStoreIndex,
    fewshot_path: str,
    tenant_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> ToolResult:
    q = str(question or "").strip()
    qn = _norm_ascii_text(q)
    ask_list_courses = any(
        k in qn
        for k in [
            "co nhung khoa hoc nao",
            "cac khoa hoc",
            "khoa hoc nao",
            "danh sach khoa hoc",
        ]
    )
    ask_recommend = any(
        k in qn
        for k in [
            "nen chon khoa nao",
            "goi y khoa",
            "tu van khoa",
        ]
    ) and ("ngan sach" in qn or "thang" in qn or "phat am" in qn)

    if ask_list_courses or ask_recommend:
        retrieval_query = "danh sach khoa hoc hoc phi" if ask_list_courses else q
        retrieved = retrieve_hybrid_contexts(
            retrieval_query,
            index,
            top_k_ctx=max(RETRIEVAL_TOP_K, 8),
            tenant_id=tenant_id,
            branch_id=branch_id,
        )
        contexts = retrieved.get("contexts", []) or []
        ctx_texts = [str(c.get("text", "")) for c in contexts if isinstance(c, dict)]
        catalog = _extract_course_catalog_from_texts(ctx_texts)
        if ask_list_courses and catalog:
            names = sorted(
                list(
                    {
                        str(v.get("name") or "").strip()
                        for v in catalog.values()
                        if str(v.get("name") or "").strip()
                    }
                )
            )
            if names:
                bullets = "\n".join([f"- {n}" for n in names[:10]])
                ans = "Dạ trung tâm hiện có các khóa chính:\n" + bullets
                return ToolResult(
                    answer=_sanitize_public_answer(ans),
                    sources=_sources_from_contexts(contexts),
                    metadata={"route": "course_search", "heuristic": "course_list"},
                    context_texts=ctx_texts,
                )

        if ask_recommend and catalog:
            budget = None
            for m in re.finditer(r"(?i)\b(\d+(?:[.,]\d+)?)\s*(tr|trieu|k|vnd|dong)\b", q):
                v = parse_money_to_vnd(m.group(0))
                if isinstance(v, int) and v > 0:
                    budget = int(v) if budget is None else max(int(budget), int(v))
            m_month = re.search(r"(?i)\b(\d+(?:[.,]\d+)?)\s*thang\b", qn)
            target_month = None
            if m_month:
                try:
                    target_month = float(m_month.group(1).replace(",", "."))
                except Exception:
                    target_month = None
            wants_pron = ("phat am" in qn) or ("pronunciation" in qn)

            cands: List[Tuple[int, str, Dict[str, object]]] = []
            for item in catalog.values():
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                dur = item.get("duration_months")
                fee = item.get("tuition_vnd")
                if isinstance(target_month, (int, float)) and isinstance(dur, (int, float)):
                    if float(dur) > float(target_month) + 0.25:
                        continue
                if isinstance(budget, int) and isinstance(fee, int):
                    if int(fee) > int(budget):
                        continue
                score = 0
                if wants_pron and ("pronunciation" in _norm_ascii_text(name) or "phat am" in _norm_ascii_text(name)):
                    score += 3
                if isinstance(dur, (int, float)) and isinstance(target_month, (int, float)):
                    score += max(0, int(10 - abs(float(dur) - float(target_month)) * 3))
                if isinstance(fee, int) and isinstance(budget, int):
                    score += max(0, int((int(budget) - int(fee)) / 1_000_000))
                cands.append((score, name, item))

            cands.sort(key=lambda x: x[0], reverse=True)
            if cands:
                _score, name, item = cands[0]
                bits: List[str] = []
                if isinstance(item.get("duration_months"), (int, float)):
                    bits.append(f"{float(item['duration_months']):g} tháng")
                if isinstance(item.get("tuition_vnd"), int):
                    bits.append(_fmt_vnd_public(int(item["tuition_vnd"])))
                detail = ", ".join(bits)
                ans = (
                    f"Dạ theo nhu cầu của anh/chị, em gợi ý **{name}**"
                    + (f" ({detail})." if detail else ".")
                )
                return ToolResult(
                    answer=_sanitize_public_answer(ans),
                    sources=_sources_from_contexts(contexts),
                    metadata={"route": "course_search", "heuristic": "course_recommend"},
                    context_texts=ctx_texts,
                )

    result = query_with_incontext_ralm(
        user_query=q,
        index=index,
        fewshot_path=fewshot_path,
        top_k_ctx=RETRIEVAL_TOP_K,
        top_k_examples=3,
        tenant_id=tenant_id,
        branch_id=branch_id,
        history=history or [],
    )
    return ToolResult(
        answer=_sanitize_public_answer(str(result.get("answer", ""))),
        sources=[str(s) for s in (result.get("sources", []) or [])],
        context_texts=[str(c) for c in (result.get("contexts", []) or [])],
    )


def tuition_calculator_tool(
    question: str,
    *,
    index: Optional[VectorStoreIndex],
    tenant_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    allow_llm: bool = False,
) -> ToolResult:
    """
    Tool-first answer: compute total payment (tuition +/- discount + extra fees such as materials)
    using direct parsing when possible; otherwise use retrieval-only then extract evidence.
    """
    q = (question or "").strip()

    def _fmt_vnd(v: int) -> str:
        return f"{int(v):,}".replace(",", ".") + " VND"

    def _strip_accents(s: str) -> str:
        s = unicodedata.normalize("NFD", s or "")
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        return s.replace("đ", "d").replace("Đ", "D")

    def _parse_discount_percent(text: str) -> Optional[float]:
        m = re.search(r"(?i)(\d{1,3}(?:[.,]\d+)?)\s*%", text or "")
        if not m:
            return None
        try:
            return float(m.group(1).replace(",", "."))
        except Exception:
            return None

    def _parse_discount_amount(text: str) -> Optional[int]:
        # e.g. "giảm 500k", "giảm 1tr", "giảm 500.000đ"
        m = re.search(r"(?i)\b(?:giảm|giam)\s+([^\n,;]{1,20})", text or "")
        if not m:
            return None
        frag = m.group(1)
        if "%" in frag:
            return None
        v = parse_money_to_vnd(frag)
        if v is not None:
            return int(v)
        mk = re.search(r"(?i)\b(\d+(?:[.,]\d+)?)\s*k\b", frag.strip())
        if mk:
            try:
                return int(round(float(mk.group(1).replace(",", ".")) * 1000))
            except Exception:
                return None
        return None

    def _parse_k_to_vnd(s: str) -> Optional[int]:
        m = re.search(r"(?i)\b(\d+(?:[.,]\d+)?)\s*k\b", (s or "").strip())
        if not m:
            return None
        try:
            return int(round(float(m.group(1).replace(",", ".")) * 1000))
        except Exception:
            return None

    def _parse_grouped_number_vnd(s: str) -> Optional[int]:
        """
        OCR fallback: parse dot-grouped numbers like 9.000.000 even when currency suffix is missing.
        Accepts variants like "IELTS: 9.000.000" or "9.000.000.0".
        """
        raw = (s or "").strip()
        m = re.search(r"(?i)(?:\b|:)\s*(\d{1,3}(?:\.\d{3})+(?:\.\d+)?)\b", raw)
        if not m:
            return None
        frag = m.group(1)
        parts = frag.split(".")
        if parts and len(parts[-1]) != 3:
            parts = parts[:-1]
        if not parts:
            return None
        num = "".join([p for p in parts if p.isdigit()])
        if len(num) < 5:
            return None
        try:
            return int(num)
        except Exception:
            return None

    def _apply_discount(base_vnd: int, *, percent: Optional[float], amount_vnd: Optional[int]) -> Optional[int]:
        if base_vnd <= 0:
            return None
        if percent is not None:
            if percent < 0 or percent > 100:
                return None
            return int(round(base_vnd * (1.0 - percent / 100.0)))
        if amount_vnd is not None:
            if amount_vnd < 0 or amount_vnd > base_vnd:
                return None
            return int(base_vnd - amount_vnd)
        return None

    def _extract_all_money_vnd(text: str) -> List[int]:
        vals: List[int] = []
        for m in re.finditer(r"(?i)(\d[\d\., ]{4,}\s*(?:₫|đ|vnd|dong)\b)", text or ""):
            v = parse_money_to_vnd(m.group(1))
            if v is not None:
                vals.append(int(v))
        for m in re.finditer(r"(?i)(\d+(?:[.,]\d+)?)\s*(tr|tri[eệ]u)\b", text or ""):
            v = parse_money_to_vnd(m.group(0))
            if v is not None:
                vals.append(int(v))
        for m in re.finditer(r"(?i)\b(\d+(?:[.,]\d+)?)\s*k\b", text or ""):
            v = _parse_k_to_vnd(m.group(0))
            if v is not None:
                vals.append(int(v))
        # OCR fallback: grouped numbers without suffix, e.g. 9.000.000
        for m in re.finditer(r"(?i)(?:\b|:)\s*(\d{1,3}(?:\.\d{3})+(?:\.\d+)?)\b", text or ""):
            v = _parse_grouped_number_vnd(m.group(0))
            if v is not None:
                vals.append(int(v))
        # De-dup
        return sorted(set([x for x in vals if isinstance(x, int) and x > 0]))

    percent_from_query = _parse_discount_percent(q)
    amount_from_query = _parse_discount_amount(q)
    suspicious_instruction = _contains_injection_cues(q)
    focus_course_from_query = _extract_focus_course_from_query(q)

    # Prefer base tuition on the left side of "giảm/giam" to avoid confusing it with discount amount.
    left = re.split(r"(?i)\b(?:giảm|giam)\b", q, maxsplit=1)[0]
    left_vals = _extract_all_money_vnd(left)
    all_vals = _extract_all_money_vnd(q)
    base_from_query: Optional[int] = max(left_vals) if left_vals else (max(all_vals) if all_vals else None)
    if amount_from_query is not None and base_from_query == int(amount_from_query) and len(all_vals) <= 1:
        # Likely the user only mentioned the discount amount (missing base tuition).
        base_from_query = None
    # Safety: do not trust user-supplied price in suspicious/instruction-like prompts.
    if suspicious_instruction and index is not None:
        base_from_query = None

    # Extra fees (materials, textbooks...) mentioned by the user.
    TUITION_HINTS = ["học phí", "hoc phi", "tuition", "gói học", "goi hoc", "trọn gói", "tron goi", "niêm yết", "niem yet"]
    FEE_HINTS = [
        "giáo trình",
        "giao trinh",
        "hoc lieu",
        "tài liệu",
        "tai lieu",
        "material",
        "phí giáo trình",
        "phi giao trinh",
        "phi hoc lieu",
        "phí tài liệu",
        "phi tai lieu",
        "phí bảo lưu",
        "phi bao luu",
        "phí học bù",
        "phi hoc bu",
        "phu phi",
        "le phi",
    ]
    TUITION_HINTS_N = [_strip_accents(x).lower() for x in TUITION_HINTS]
    FEE_HINTS_N = [_strip_accents(x).lower() for x in FEE_HINTS]

    def _iter_money_mentions(text: str) -> List[Tuple[int, int, int]]:
        out: List[Tuple[int, int, int]] = []
        tt = text or ""
        for m in re.finditer(r"(?i)(\d[\d\., ]{4,}\s*(?:₫|đ|vnd|dong)\b)", tt):
            v = parse_money_to_vnd(m.group(1))
            if v is not None:
                out.append((int(v), m.start(1), m.end(1)))
        for m in re.finditer(r"(?i)\b(\d+(?:[.,]\d+)?)\s*(tr|tri[eệ]u)\b", tt):
            v = parse_money_to_vnd(m.group(0))
            if v is not None:
                out.append((int(v), m.start(0), m.end(0)))
        for m in re.finditer(r"(?i)\b(\d+(?:[.,]\d+)?)\s*k\b", tt):
            v = _parse_k_to_vnd(m.group(0))
            if v is not None:
                out.append((int(v), m.start(0), m.end(0)))
        for m in re.finditer(r"(?i)(?:\b|:)\s*(\d{1,3}(?:\.\d{3})+(?:\.\d+)?)\b", tt):
            v = _parse_grouped_number_vnd(m.group(0))
            if v is not None:
                out.append((int(v), m.start(1), m.end(1)))
        return out

    def _around(text: str, start: int, end: int, win: int = 60) -> str:
        return (text or "")[max(0, start - win) : min(len(text or ""), end + win)]

    left_mentions = _iter_money_mentions(left)
    # Prefer the amount explicitly labeled as tuition if present.
    tuition_labeled = []
    for v, s, e in left_mentions:
        a = _strip_accents(_around(left, s, e)).lower()
        if any(k in a for k in TUITION_HINTS_N):
            tuition_labeled.append(v)
    if tuition_labeled:
        base_from_query = max(tuition_labeled)

    # Extra fees can appear before or after the discount phrase; scan the whole query but ignore discount amounts.
    def _discount_amount_spans(text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        for m in re.finditer(r"(?i)\b(?:giảm|giam)\s+([^\n,;]{1,24})", text or ""):
            frag = m.group(1) or ""
            if "%" in frag:
                continue
            spans.append((m.start(1), m.end(1)))
        return spans

    def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])

    discount_spans = _discount_amount_spans(q)
    fee_total_from_query = 0
    fee_mentions_debug: List[Tuple[int, str]] = []
    for v, s, e in _iter_money_mentions(q):
        if base_from_query is not None and int(v) == int(base_from_query):
            continue
        if amount_from_query is not None and int(v) == int(amount_from_query) and any(_overlaps((s, e), sp) for sp in discount_spans):
            continue
        if any(_overlaps((s, e), sp) for sp in discount_spans):
            continue
        a = _strip_accents(_around(q, s, e)).lower()
        if any(k in a for k in FEE_HINTS_N):
            fee_total_from_query += int(v)
            fee_mentions_debug.append((int(v), a[:80]))

    q_norm = _strip_accents(q).lower()
    wants_discount_calc = bool("giam" in q_norm or "%" in q_norm or "sau giam" in q_norm or "discount" in q_norm)
    ask_non_stack_policy = ("cong don" in q_norm) and (
        ("giam" in q_norm) or ("phan tram" in q_norm) or ("uu dai" in q_norm)
    )
    asks_bao_luu = ("bao luu" in q_norm) or ("phi bao luu" in q_norm)
    combo_mode_rough = ("combo" in q_norm) and ("khoa" in q_norm)

    if suspicious_instruction:
        return ToolResult(
            answer=(
                "Dạ em chỉ có thể tư vấn dựa trên tài liệu chính thức của trung tâm và "
                "không thể làm theo yêu cầu bỏ qua tài liệu. Anh/chị cho em biết khóa học cụ thể "
                "trong trung tâm để em báo học phí chính xác ạ."
            ),
            sources=[],
            metadata={
                "route": "tuition_calculator",
                "calc_mode": "blocked_injection",
                "requested_course": focus_course_from_query,
            },
        )

    def _detect_discount_scope(q_norm_s: str) -> str:
        """
        Decide whether discount applies to:
        - "tuition": only the tuition/package fee (default)
        - "total": total payment (tuition + extra fees)
        Rules (per product requirement):
        - "giảm ... gói học" => tuition-only
        - "giảm tổng chi phí/tổng tiền/tổng thanh toán/giảm tổng" => total
        """
        s = (q_norm_s or "").lower()
        total_patterns = [
            r"\bgiam\s+tong\b",
            r"\bgiam\s+tren\s+tong\b",
            r"\bgiam\s+toan\s+bo\b",
            r"\bgiam\s+\d+(?:[.,]\d+)?%\s+tren\s+tong\b",
        ]
        if any(re.search(p, s) for p in total_patterns):
            return "total"
        # Explicitly treat "gói học" as tuition-only unless user says "tổng ..."
        if "goi hoc" in s or "hoc phi" in s:
            return "tuition"
        return "tuition"

    discount_scope = _detect_discount_scope(q_norm)

    # Optional: if we see multiple money mentions but fail to attribute fees, ask LLM to classify.
    llm_fee_extraction: Dict[str, object] | None = None
    if allow_llm and base_from_query is not None:
        all_mentions = _extract_all_money_vnd(q)
        if fee_total_from_query == 0 and len(all_mentions) >= 2:
            try:
                llm_fee_extraction_raw = extract_financials_with_llm(text=q, question=q)
                _b, _extra, _refined = refine_extracted_fees(llm_fee_extraction_raw, course_name_query=q)
                llm_fee_extraction = _refined
                if int(_extra) > 0:
                    fee_total_from_query = int(_extra)
            except Exception:
                llm_fee_extraction = None

    # Direct calculation mode (no retrieval) when user provides enough inputs.
    if wants_discount_calc and base_from_query is not None and (percent_from_query is not None or amount_from_query is not None):
        total_before_discount_vnd = int(base_from_query) + int(fee_total_from_query or 0)
        tuition_after_discount_vnd = None
        total_after_discount_vnd = None

        if discount_scope == "total":
            total_after_discount_vnd = _apply_discount(
                int(total_before_discount_vnd),
                percent=percent_from_query,
                amount_vnd=amount_from_query,
            )
        else:
            tuition_after_discount_vnd = _apply_discount(
                int(base_from_query),
                percent=percent_from_query,
                amount_vnd=amount_from_query,
            )
            if tuition_after_discount_vnd is not None:
                total_after_discount_vnd = int(tuition_after_discount_vnd) + int(fee_total_from_query or 0)

        if total_after_discount_vnd is None:
            return ToolResult(
                answer="Dạ em chưa thể tính vì mức giảm không hợp lệ (ví dụ % > 100 hoặc số tiền giảm lớn hơn học phí). Anh/chị kiểm tra lại giúp em nhé.",
                sources=[],
                metadata={
                    "route": "tuition_calculator",
                    "calc_mode": "direct",
                    "computed_final_vnd": None,
                    "computed_tuition_after_discount_vnd": None,
                    "extra_fees_vnd": int(fee_total_from_query or 0),
                    "discount_scope": discount_scope,
                    "discount": {
                        "base_from_query_vnd": base_from_query,
                        "percent_from_query": percent_from_query,
                        "amount_from_query_vnd": amount_from_query,
                    },
                },
            )
        discount_desc = (
            f"giảm {percent_from_query:g}%"
            if percent_from_query is not None
            else f"giảm **{_fmt_vnd(int(amount_from_query or 0))}**"
        )
        breakdown = ""
        if discount_scope == "total":
            breakdown = f" (tổng trước giảm **{_fmt_vnd(int(total_before_discount_vnd))}**)"
        elif fee_total_from_query and tuition_after_discount_vnd is not None:
            breakdown = (
                f" (học phí sau giảm **{_fmt_vnd(int(tuition_after_discount_vnd))}**"
                f" + phí **{_fmt_vnd(int(fee_total_from_query))}**)"
            )

        # If user has fees but didn't specify scope, keep default (tuition) and mention the alternative.
        note = ""
        if fee_total_from_query and discount_scope == "tuition" and not any(k in q_norm for k in ["goi hoc", "hoc phi", "giam tong", "tong chi phi", "tong tien", "tong thanh toan", "tong cong"]):
            note = " (Nếu giảm áp dụng trên **tổng** thì hãy nói rõ: “giảm tổng chi phí …”.)"

        return ToolResult(
            answer=f"Dạ tổng thanh toán sau {discount_desc} là **{_fmt_vnd(int(total_after_discount_vnd))}**{breakdown}.{note}",
            sources=[],
            metadata={
                "route": "tuition_calculator",
                "calc_mode": "direct",
                "computed_final_vnd": int(total_after_discount_vnd),
                "computed_tuition_after_discount_vnd": int(tuition_after_discount_vnd) if tuition_after_discount_vnd is not None else None,
                "extra_fees_vnd": int(fee_total_from_query or 0),
                "fee_extraction": {
                    "discount_amount_spans": discount_spans,
                    "fee_mentions": [(v, snip) for v, snip in fee_mentions_debug[:6]],
                    "llm_used": bool(llm_fee_extraction is not None),
                },
                "discount_scope": discount_scope,
                "total_before_discount_vnd": int(total_before_discount_vnd),
                "discount": {
                    "base_from_query_vnd": base_from_query,
                    "percent_from_query": percent_from_query,
                    "amount_from_query_vnd": amount_from_query,
                },
            },
        )

    if wants_discount_calc and base_from_query is not None and percent_from_query is None and amount_from_query is None:
        return ToolResult(
            answer="Dạ anh/chị cho em xin mức giảm cụ thể (ví dụ 10% hoặc giảm 500k) để em tính học phí sau giảm ạ.",
            sources=[],
            metadata={"route": "tuition_calculator", "calc_mode": "direct_need_discount", "base_from_query_vnd": base_from_query},
        )

    if index is None:
        return ToolResult(
            answer="Dạ anh/chị cho em biết khóa/gói đang quan tâm (IELTS/TOEIC/Giao tiếp/Thiếu nhi) để em tra đúng học phí và ưu đãi trong tài liệu ạ.",
            sources=[],
            metadata={"route": "tuition_calculator", "calc_mode": "no_index"},
        )

    retrieval_query = question
    if ask_non_stack_policy:
        retrieval_query = "chinh sach uu dai khong duoc cong don dang ky nhom dang ky som"
    elif combo_mode_rough and asks_bao_luu:
        retrieval_query = "combo 2 khoa hoc phi phi bao luu chinh sach bao luu"
    elif focus_course_from_query:
        retrieval_query = f"{focus_course_from_query} hoc phi phi tai lieu uu dai"
    retrieved = retrieve_hybrid_contexts(
        retrieval_query,
        index,
        top_k_ctx=max(RETRIEVAL_TOP_K, 10 if focus_course_from_query else RETRIEVAL_TOP_K),
        tenant_id=tenant_id,
        branch_id=branch_id,
    )
    contexts = retrieved.get("contexts", []) or []
    if not isinstance(contexts, list) or not contexts:
        return ToolResult(
            answer="Dạ em chưa tìm thấy thông tin học phí phù hợp trong tài liệu hiện có. Anh/chị cho em biết mình đang quan tâm khóa nào (IELTS/TOEIC/Giao tiếp/Thiếu nhi) ạ?",
            sources=[],
            metadata={"route": "tuition_calculator"},
        )

    # Secondary retrieval boost for policy/hold-fee questions to avoid missing policy nodes.
    if ask_non_stack_policy or asks_bao_luu:
        extra_query_parts: List[str] = []
        if ask_non_stack_policy:
            extra_query_parts.extend(["chinh sach uu dai", "khong duoc cong don", "khong cong don"])
        if asks_bao_luu:
            extra_query_parts.extend(["chinh sach bao luu", "phi bao luu"])
        if focus_course_from_query:
            extra_query_parts.append(focus_course_from_query)
        extra_query = " ".join(extra_query_parts).strip() or question
        try:
            extra_retrieved = retrieve_hybrid_contexts(
                extra_query,
                index,
                top_k_ctx=max(RETRIEVAL_TOP_K, 8),
                tenant_id=tenant_id,
                branch_id=branch_id,
            )
            extra_contexts = extra_retrieved.get("contexts", []) or []
            if isinstance(extra_contexts, list) and extra_contexts:
                contexts = _merge_context_dicts(contexts, extra_contexts)
        except Exception:
            pass

    combined = "\n\n".join([str(c.get("text", "")) for c in contexts if isinstance(c, dict)])
    combined_norm = _norm_ascii_text(combined)
    has_grouped_nums = bool(re.search(r"(?i)(?:\b|:)\s*(\d{1,3}(?:\.\d{3})+(?:\.\d+)?)\b", combined))
    evidence = extract_evidence_dict(combined)
    course_price_map = _extract_course_fee_pairs_from_text(combined)
    matched_course_fee = (
        _match_course_price(course_hint=focus_course_from_query, course_price_map=course_price_map)
        if focus_course_from_query
        else None
    )

    # Fail-safe: if user asks a specific course but this tenant context does not contain it, do not hallucinate.
    if focus_course_from_query and not _text_mentions_course(course=focus_course_from_query, text=combined):
        no_match = (
            "Dạ hiện tại em chưa thấy khóa này trong tài liệu của trung tâm. "
            "Anh/chị giúp em xác nhận đúng tên khóa hoặc cho em biết khóa đang quan tâm trong trung tâm hiện tại để em báo học phí chính xác ạ."
        )
        return ToolResult(
            answer=no_match,
            sources=_sources_from_contexts(contexts),
            metadata={
                "route": "tuition_calculator",
                "calc_mode": "course_not_found",
                "requested_course": focus_course_from_query,
            },
            context_texts=[str(c.get("text", "")) for c in contexts],
        )

    # "Mắt thần" (LLM extractor): always attempt finance classification on retrieved context
    # when LLM usage is allowed. This reduces confusion between fees (e.g. 300k) and tuition (e.g. 9m).
    llm_fin_ctx: Dict[str, object] | None = None
    if allow_llm and combined.strip():
        try:
            llm_fin_ctx_raw = extract_financials_with_llm(text=combined[:8000], question=q)
            _, _, llm_fin_ctx = refine_extracted_fees(llm_fin_ctx_raw, course_name_query=q)
        except Exception:
            llm_fin_ctx = None

    # Fallback: if user asks for discount but didn't provide it, try to find in retrieved context.
    percent_from_ctx = _parse_discount_percent(combined)
    amount_from_ctx = _parse_discount_amount(combined)
    policy_discount_percents: List[float] = []
    for m in re.finditer(r"(?i)giam[^\n]{0,40}?(\d{1,2}(?:[.,]\d+)?)\s*%", combined):
        try:
            policy_discount_percents.append(float(m.group(1).replace(",", ".")))
        except Exception:
            continue
    if not policy_discount_percents:
        policy_discount_percents = _extract_discount_percents_from_text(combined)
    policy_discount_percents = [x for x in policy_discount_percents if 0.0 < x <= 50.0]
    policy_discount_percents = sorted(set([x for x in policy_discount_percents if 0.0 < x <= 50.0]))

    if ask_non_stack_policy and policy_discount_percents:
        best = max(policy_discount_percents)
        has_non_stack_rule = ("khong cong don" in combined_norm) or ("khong duoc cong don" in combined_norm)
        if not has_non_stack_rule:
            has_non_stack_rule = policy_registry.has_non_stack_rule(str(tenant_id or ""))
        if has_non_stack_rule:
            ans = (
                f"Dạ chính sách ưu đãi **không cộng dồn**. "
                f"Nếu đồng thời thỏa nhiều ưu đãi, trung tâm áp dụng mức cao nhất là **{best:g}%** ạ."
            )
        else:
            ans = (
                f"Dạ em thấy các mức ưu đãi tối đa đến **{best:g}%**. "
                "Anh/chị giúp em xác nhận thêm chính sách cộng dồn để em chốt chính xác ạ."
            )
        return ToolResult(
            answer=ans,
            sources=_sources_from_contexts(contexts),
            metadata={
                "route": "tuition_calculator",
                "calc_mode": "policy_non_stack",
                "best_discount_percent": float(best),
                "non_stack_detected": bool(has_non_stack_rule),
            },
            context_texts=[str(c.get("text", "")) for c in contexts],
        )

    tuition_vals = [int(x) for x in evidence.get("tuition_vnd", []) if isinstance(x, (int, float))]
    fees_vals = [int(x) for x in evidence.get("fees_vnd", []) if isinstance(x, (int, float))]
    duration_months = [float(x) for x in evidence.get("duration_months", []) if isinstance(x, (int, float))]

    # OCR fallback: classify dot-grouped numbers (e.g. 9.000.000) when regex-evidence misses them.
    if has_grouped_nums and (not tuition_vals or not fees_vals):
        for v, s, e in _iter_money_mentions(combined):
            around = _strip_accents(_around(combined, s, e)).lower()
            if v < 1_000_000 and not any(k in around for k in TUITION_HINTS_N):
                fees_vals.append(int(v))
            elif any(k in around for k in FEE_HINTS_N) and not any(k in around for k in TUITION_HINTS_N):
                fees_vals.append(int(v))
            else:
                tuition_vals.append(int(v))
        tuition_vals = sorted(set([int(x) for x in tuition_vals if int(x) > 0]))
        fees_vals = sorted(set([int(x) for x in fees_vals if int(x) > 0]))

    # Prefer LLM-extracted base/fees from context when present.
    llm_base_ctx = int(llm_fin_ctx.get("base_tuition_vnd") or 0) if isinstance(llm_fin_ctx, dict) else 0
    llm_extra_ctx = int(llm_fin_ctx.get("extra_fees_vnd") or 0) if isinstance(llm_fin_ctx, dict) else 0
    if llm_base_ctx >= 1_000_000:
        tuition_vals = sorted(set(tuition_vals + [llm_base_ctx]))
    if llm_extra_ctx > 0:
        fees_vals = sorted(set(fees_vals + [llm_extra_ctx]))

    # Extra fees requested by user: prefer explicit amounts in the query; otherwise use small fees from evidence.
    wants_total_payment = bool(
        re.search(r"\btong\b", q_norm)
        or any(k in q_norm for k in ["tong chi phi", "tong tien", "tong thanh toan", "tong cong"])
    )
    extra_fee_terms = [
        "giao trinh",
        "tai lieu",
        "hoc lieu",
        "material",
        "phi giao trinh",
        "phi tai lieu",
        "phi hoc lieu",
        "phi bao luu",
        "phi hoc bu",
        "phu phi",
        "le phi",
    ]
    wants_materials = bool(fee_total_from_query) or wants_total_payment or any(k in q_norm for k in extra_fee_terms)
    extra_fees_vnd = int(fee_total_from_query or 0)
    if wants_materials and extra_fees_vnd == 0:
        label_cues: List[str] = []
        if "tai lieu" in q_norm or "giao trinh" in q_norm or "hoc lieu" in q_norm:
            label_cues.extend(["phi tai lieu", "phi giao trinh", "phi hoc lieu"])
        if "bao luu" in q_norm:
            label_cues.append("phi bao luu")
        if "hoc bu" in q_norm:
            label_cues.append("phi hoc bu")
        explicit_fee = _extract_fee_by_label_from_text(text=combined, labels=label_cues)
        if isinstance(explicit_fee, int) and explicit_fee > 0:
            extra_fees_vnd = int(explicit_fee)
    if asks_bao_luu and int(fee_total_from_query or 0) == 0:
        hold_fee = _extract_fee_by_label_from_text(text=combined, labels=["phi bao luu", "bao luu"])
        if isinstance(hold_fee, int) and hold_fee > 0:
            extra_fees_vnd = int(hold_fee)
    if wants_materials and extra_fees_vnd == 0 and llm_extra_ctx > 0:
        extra_fees_vnd = int(llm_extra_ctx)
    if wants_materials and extra_fees_vnd == 0 and fees_vals:
        # Keep conservative: only sum small fees (avoid mixing unrelated deposits/refunds).
        fee_set: set[int] = set()
        for v in fees_vals:
            iv = int(v)
            if 0 < iv < 2_000_000:
                if (iv % 1000) != 0:
                    iv = int(round(iv / 1000.0) * 1000)
                fee_set.add(iv)
        extra_fees_vnd = int(sum(sorted(fee_set)))

    # Pick base tuition candidates
    base_candidates: List[int] = []
    if isinstance(matched_course_fee, int) and matched_course_fee >= 1_000_000:
        base_candidates = [int(matched_course_fee)]
    elif llm_base_ctx >= 1_000_000:
        base_candidates = [int(llm_base_ctx)]
    elif tuition_vals:
        base_candidates = sorted(set(tuition_vals))
    elif base_from_query is not None:
        # Only trust query-supplied base when retrieval could not resolve tuition.
        allow_query_base = (index is None) or (not suspicious_instruction and not focus_course_from_query)
        if allow_query_base:
            base_candidates = [int(base_from_query)]
    if focus_course_from_query and matched_course_fee is None and index is not None:
        # Avoid mapping a specific requested course to unrelated tuition values.
        base_candidates = []

    # Pick discount inputs (prefer query)
    discount_percent = percent_from_query if percent_from_query is not None else None
    discount_amount = amount_from_query if amount_from_query is not None else None
    if wants_discount_calc and discount_percent is None and discount_amount is None:
        discount_percent = percent_from_ctx
        discount_amount = amount_from_ctx

    # Combo flow: sum explicit course fees from context, then apply combo discount.
    combo_mode = ("combo" in q_norm) and ("khoa" in q_norm)
    if combo_mode and course_price_map:
        mentioned: List[Tuple[int, str, int]] = []
        for ck, cv in course_price_map.items():
            toks = [t for t in ck.split() if len(t) >= 4]
            if not toks:
                continue
            hits = sum(1 for t in toks if re.search(rf"\b{re.escape(t)}\b", q_norm))
            if hits >= min(2, len(toks)):
                mentioned.append((hits, ck, int(cv)))
        mentioned.sort(key=lambda x: (x[0], x[2]), reverse=True)
        chosen = mentioned[:2]
        if len(chosen) >= 2:
            combo_base = int(chosen[0][2]) + int(chosen[1][2])
            combo_percent = discount_percent
            if combo_percent is None:
                combo_percent = _extract_combo_discount_percent(combined)
            if combo_percent is not None and not (0.0 < float(combo_percent) <= 50.0):
                combo_percent = None

            tuition_after = combo_base
            if combo_percent is not None:
                maybe_after = _apply_discount(combo_base, percent=combo_percent, amount_vnd=None)
                if maybe_after is not None:
                    tuition_after = int(maybe_after)
            total_after = int(tuition_after) + int(extra_fees_vnd or 0)
            chosen_names = [str(x[1]).title() for x in chosen]
            answer = (
                f"Dạ tổng thanh toán cho combo ({chosen_names[0]} + {chosen_names[1]}) là **{_fmt_vnd(total_after)}**."
            )
            if combo_percent is not None:
                answer += f" (Đã áp dụng giảm combo {combo_percent:g}%.)"
            if int(extra_fees_vnd or 0) > 0:
                answer += f" (Đã gồm phí {_fmt_vnd(int(extra_fees_vnd))}.)"
            return ToolResult(
                answer=answer,
                sources=_sources_from_contexts(contexts),
                metadata={
                    "route": "tuition_calculator",
                    "calc_mode": "combo_context",
                    "computed_final_vnd": int(total_after),
                    "computed_tuition_after_discount_vnd": int(tuition_after),
                    "computed_range_vnd": None,
                    "extra_fees_vnd": int(extra_fees_vnd or 0),
                    "discount_scope": "tuition",
                    "discount": {
                        "base_from_query_vnd": None,
                        "percent_from_query": combo_percent,
                        "amount_from_query_vnd": None,
                    },
                    "combo_courses": chosen_names,
                },
                context_texts=[str(c.get("text", "")) for c in contexts],
            )

    parts: List[str] = []
    if base_candidates:
        mn, mx = min(base_candidates), max(base_candidates)
        parts.append(f"học phí khoảng {_fmt_vnd(mn)}" if mn == mx else f"học phí khoảng {_fmt_vnd(mn)}–{_fmt_vnd(mx)}")
    if duration_months:
        dm = sorted(set(duration_months))
        parts.append(f"thời lượng {dm[0]:g} tháng" if len(dm) == 1 else f"thời lượng {dm[0]:g}–{dm[-1]:g} tháng")

    finals: List[int] = []
    if wants_discount_calc:
        if not base_candidates:
            answer = (
                "Dạ để tính học phí sau giảm, anh/chị cho em xin (1) khóa/gói đang quan tâm hoặc học phí gốc, "
                "và (2) mức giảm (% hoặc số tiền) ạ."
            )
        elif discount_percent is None and discount_amount is None:
            answer = (
                "Dạ anh/chị đang muốn tính học phí sau giảm đúng không ạ? "
                "Anh/chị cho em xin mức giảm cụ thể (ví dụ 10% hoặc giảm 500k) để em tính chính xác."
            )
        else:
            for b in base_candidates:
                if discount_scope == "total":
                    v = _apply_discount(int(b) + int(extra_fees_vnd or 0), percent=discount_percent, amount_vnd=discount_amount)
                else:
                    v = _apply_discount(int(b), percent=discount_percent, amount_vnd=discount_amount)
                if v is not None:
                    finals.append(v)
            finals = sorted(set(finals))
            if not finals:
                answer = "Dạ em chưa thể tính vì mức giảm không hợp lệ (ví dụ % > 100 hoặc số tiền giảm lớn hơn học phí). Anh/chị kiểm tra lại giúp em nhé."
            else:
                mnf, mxf = min(finals), max(finals)
                if discount_percent is not None:
                    discount_desc = f"giảm {discount_percent:g}%"
                else:
                    discount_desc = f"giảm {_fmt_vnd(int(discount_amount or 0))}"

                if mnf == mxf:
                    if discount_scope == "total":
                        calc_line = f"Tổng thanh toán sau {discount_desc}: **{_fmt_vnd(mnf)}** (đã giảm trên tổng)."
                    else:
                        if extra_fees_vnd:
                            total = int(mnf) + int(extra_fees_vnd)
                            calc_line = (
                                f"Tổng thanh toán sau {discount_desc}: **{_fmt_vnd(total)}** "
                                f"(học phí sau giảm {_fmt_vnd(mnf)} + phí {_fmt_vnd(extra_fees_vnd)})."
                            )
                        else:
                            calc_line = f"Tổng thanh toán sau {discount_desc}: **{_fmt_vnd(mnf)}**."
                else:
                    if discount_scope == "total":
                        calc_line = f"Tổng thanh toán sau {discount_desc}: khoảng **{_fmt_vnd(mnf)}–{_fmt_vnd(mxf)}** (đã giảm trên tổng)."
                    else:
                        if extra_fees_vnd:
                            tmin = int(mnf) + int(extra_fees_vnd)
                            tmax = int(mxf) + int(extra_fees_vnd)
                            calc_line = f"Tổng thanh toán sau {discount_desc}: khoảng **{_fmt_vnd(tmin)}–{_fmt_vnd(tmax)}** (đã gồm phí {_fmt_vnd(extra_fees_vnd)})."
                        else:
                            calc_line = f"Tổng thanh toán sau {discount_desc}: khoảng **{_fmt_vnd(mnf)}–{_fmt_vnd(mxf)}**."
                extra = ""
                if len(base_candidates) > 1 and base_from_query is None:
                    extra = " (Do tài liệu có nhiều mức học phí khác nhau theo gói/đầu vào; anh/chị cho em biết khóa/gói cụ thể để chốt đúng mức ạ.)"
                answer = "Dạ " + calc_line + extra
    else:
        if parts:
            if extra_fees_vnd and base_candidates:
                mn, mx = min(base_candidates), max(base_candidates)
                tmin = int(mn) + int(extra_fees_vnd)
                tmax = int(mx) + int(extra_fees_vnd)
                tot = f"tổng thanh toán khoảng {_fmt_vnd(tmin)}" if tmin == tmax else f"tổng thanh toán khoảng {_fmt_vnd(tmin)}–{_fmt_vnd(tmax)}"
                answer = (
                    "Dạ theo tài liệu em tìm được, " + tot + f" (học phí + phí {_fmt_vnd(extra_fees_vnd)}). "
                    "Anh/chị đang quan tâm khóa nào cụ thể để em báo đúng gói/ưu đãi (nếu có) ạ?"
                )
            else:
                answer = "Dạ theo tài liệu em tìm được, " + " và ".join(parts) + ". Anh/chị đang quan tâm khóa nào cụ thể để em báo đúng gói/ưu đãi (nếu có) ạ?"
        else:
            answer = "Dạ em có tìm thấy các đoạn liên quan đến học phí, nhưng chưa đủ rõ để chốt con số chính xác. Anh/chị cho em biết khóa mình quan tâm và trình độ hiện tại được không ạ?"

    computed_tuition_after_discount_vnd = None
    computed_final_vnd = None
    computed_range_vnd = None
    if finals:
        if min(finals) == max(finals):
            if discount_scope == "total":
                computed_final_vnd = int(min(finals))
            else:
                computed_tuition_after_discount_vnd = int(min(finals))
                computed_final_vnd = int(min(finals)) + int(extra_fees_vnd or 0)
        else:
            if discount_scope == "total":
                computed_range_vnd = [int(min(finals)), int(max(finals))]
            else:
                computed_range_vnd = [int(min(finals)) + int(extra_fees_vnd or 0), int(max(finals)) + int(extra_fees_vnd or 0)]
    elif (not wants_discount_calc) and base_candidates and extra_fees_vnd:
        if min(base_candidates) == max(base_candidates):
            computed_final_vnd = int(min(base_candidates)) + int(extra_fees_vnd)
        else:
            computed_range_vnd = [int(min(base_candidates)) + int(extra_fees_vnd), int(max(base_candidates)) + int(extra_fees_vnd)]

    return ToolResult(
        answer=answer,
        sources=_sources_from_contexts(contexts),
        metadata={
            "route": "tuition_calculator",
            "evidence": evidence,
            "fee_extraction": {
                "has_grouped_numbers": bool(has_grouped_nums),
                "llm_used": bool(llm_fin_ctx is not None),
                "llm_ctx": llm_fin_ctx,
            },
            "retrieval_metrics": retrieved.get("metrics"),
            "computed_final_vnd": computed_final_vnd,
            "computed_range_vnd": computed_range_vnd,
            "computed_tuition_after_discount_vnd": computed_tuition_after_discount_vnd,
            "extra_fees_vnd": int(extra_fees_vnd or 0),
            "discount_scope": discount_scope,
            "discount": {
                "base_from_query_vnd": base_from_query,
                "percent_from_query": percent_from_query,
                "amount_from_query_vnd": amount_from_query,
            },
        },
        context_texts=[str(c.get("text", "")) for c in contexts]
    )


def _parse_compare_entities(query: str) -> List[str]:
    q = (query or "").strip()

    def clean_commands(s: str) -> str:
        # Remove CLI-like flags embedded inside the query, e.g.:
        #   "... /tenant flexenglish", "... /index on", "... /llm off"
        # Only match when the slash command starts at BOF or after whitespace.
        s = re.sub(r"(?i)(?:^|\s)/(?:tenant|branch|index|llm|state|help|exit)\b(?:\s+\S+)?", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def norm(s: str) -> str:
        s = unicodedata.normalize("NFD", s or "")
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        s = s.replace("đ", "d").replace("Đ", "D").lower()
        s = re.sub(r"[^a-z0-9]+", " ", s).strip()
        return s

    q_clean = clean_commands(q)
    qn = norm(q_clean)

    # Prefer strong comparison cues first.
    if re.search(r"\bvs\b", qn):
        parts = re.split(r"\bvs\b", qn, maxsplit=1)
    elif "so sanh" in qn and re.search(r"\bvoi\b", qn):
        parts = re.split(r"\bvoi\b", qn, maxsplit=1)
    elif "so sanh" in qn and re.search(r"\bva\b", qn):
        parts = re.split(r"\bva\b", qn, maxsplit=1)
    else:
        return []

    cleaned = [p.strip(" -:\t") for p in parts if p and p.strip()]
    cleaned = [p.replace("so sanh", "").strip() for p in cleaned]
    cleaned = [p for p in cleaned if p]
    return cleaned[:2]


def comparison_tool(
    question: str,
    *,
    index: Optional[VectorStoreIndex],
    tenant_id: Optional[str] = None,
    branch_id: Optional[str] = None,
) -> ToolResult:
    entities = _parse_compare_entities(question)
    if len(entities) < 2:
        return ToolResult(
            answer="Dạ anh/chị muốn so sánh 2 khóa nào ạ? Ví dụ: “IELTS Foundation vs IELTS 6.5” hoặc “Giao tiếp vs TOEIC”.",
            sources=[],
            metadata={"route": "comparison"},
        )

    def _norm_for_match(s: str) -> str:
        s = unicodedata.normalize("NFD", s or "")
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        s = s.replace("đ", "d").replace("Đ", "D").lower()
        s = re.sub(r"[^a-z0-9]+", " ", s).strip()
        return s

    def _entity_mentioned(entity: str, blob: str) -> bool:
        ent = _norm_for_match(entity)
        txt = _norm_for_match(blob)
        if not ent or not txt:
            return False
        tokens = [t for t in ent.split() if len(t) >= 3]
        return any(re.search(rf"\b{re.escape(t)}\b", txt) for t in tokens)

    summaries: List[Tuple[str, Dict[str, object], List[Dict[str, object]], bool]] = []
    all_sources: List[str] = []
    for name in entities:
        q = f"{name} học phí thời lượng mục tiêu"
        if index is not None:
            retrieved = retrieve_hybrid_contexts(
                q,
                index,
                top_k_ctx=RETRIEVAL_TOP_K,
                tenant_id=tenant_id,
                branch_id=branch_id,
            )
            contexts = retrieved.get("contexts", []) or []
        else:
            # Offline/no-index fallback: lexical BM25 over cached nodes (fail-closed for tenant isolation).
            from app.services.retrieval.bm25 import bm25_retrieve

            contexts = bm25_retrieve(q, top_k=RETRIEVAL_TOP_K, tenant_id=tenant_id, branch_id=branch_id)
        combined = "\n\n".join([str(c.get("text", "")) for c in contexts if isinstance(c, dict)])
        present = _entity_mentioned(name, combined)
        ev = extract_evidence_dict(combined) if present else {}
        summaries.append((name, ev, contexts, present))
        all_sources.extend(_sources_from_contexts(contexts))

    # Build a compact comparison
    lines = ["Dạ em so sánh nhanh theo thông tin tìm được trong tài liệu:"]
    found: List[str] = []
    missing: List[str] = []
    for name, ev, _ctx, present in summaries:
        if not present:
            missing.append(name)
            continue
        tuition = ev.get("tuition_vnd", [])
        duration = ev.get("duration_months", [])
        ielts = ev.get("ielts_target", [])
        toeic = ev.get("toeic_target", [])
        cefr = ev.get("cefr_target", [])
        bits = []
        if tuition:
            vals = [int(x) for x in tuition if isinstance(x, (int, float))]
            if vals:
                mn, mx = min(vals), max(vals)
                bits.append((f"{mn:,}–{mx:,} VND" if mn != mx else f"{mn:,} VND").replace(",", "."))
        if duration:
            vals = [float(x) for x in duration if isinstance(x, (int, float))]
            if vals:
                mn, mx = min(vals), max(vals)
                bits.append((f"{mn:g}–{mx:g} tháng" if mn != mx else f"{mn:g} tháng"))
        if ielts:
            bits.append("IELTS " + ", ".join([str(x) for x in ielts]))
        if toeic:
            bits.append("TOEIC " + ", ".join([str(x) for x in toeic]))
        if cefr:
            bits.append("CEFR " + ", ".join([str(x) for x in cefr]))
        detail = "; ".join(bits) if bits else "chưa đủ dữ liệu rõ ràng (cần xác nhận thêm)"
        if bits:
            found.append(name)
            lines.append(f"- {name}: {detail}")
        else:
            missing.append(name)
    if not found:
        return ToolResult(
            answer="Dạ em chưa tìm thấy thông tin phù hợp cho cả 2 nội dung anh/chị muốn so sánh trong tài liệu/DB hiện có. Anh/chị cho em xin tên khóa cụ thể hoặc trung tâm/chi nhánh ạ.",
            sources=[],
            metadata={
                "route": "comparison",
                "retrieval": ("hybrid" if index is not None else "bm25_only"),
                "missing_entities": missing,
            },
        )

    if missing:
        lines.append(
            f"Dạ em tìm thấy thông tin cho {', '.join(found)}, còn {', '.join(missing)} hiện em chưa thấy trong tài liệu/DB."
        )

    lines.append("Anh/chị cho em biết mục tiêu và thời gian rảnh để em gợi ý khóa phù hợp nhất ạ.")

    # Dedup sources
    dedup_sources = []
    seen = set()
    for s in all_sources:
        if s in seen:
            continue
        seen.add(s)
        dedup_sources.append(s)

    # Collect all contexts from all entities
    all_contexts: List[str] = []
    for _name, _ev, _ctxs, _ in summaries:
         all_contexts.extend([str(c.get("text", "")) for c in _ctxs])

    return ToolResult(
        answer="\n".join(lines),
        sources=dedup_sources,
        metadata={"route": "comparison", "retrieval": ("hybrid" if index is not None else "bm25_only")},
        context_texts=all_contexts
    )


def create_ticket_tool(
    question: str,
    *,
    tenant_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> ToolResult:
    """
    Minimal ticket/handoff:
    - Best-effort write to Postgres (handoff_tickets) for dashboard metrics.
    - Also append to local JSONL for backward compatibility.
    """
    phone = extract_phone(question or "")
    if not phone:
        return ToolResult(
            answer="Dạ được ạ. Anh/chị cho em xin **SĐT** và **khung giờ thuận tiện** để tư vấn viên liên hệ hỗ trợ chi tiết nhé.",
            sources=[],
            metadata={"route": "create_ticket", "ticket_created": False},
        )

    ticket_id = None
    try:
        from app.services.analytics.store import insert_handoff_ticket

        ticket_id = insert_handoff_ticket(
            tenant_id=tenant_id,
            branch_id=branch_id,
            user_id=user_id,
            phone=phone,
            message=(question or "").strip(),
            status="new",
            meta={"source": "create_ticket_tool"},
        )
    except Exception:
        ticket_id = None

    ticket = {
        "ts": int(time.time()),
        "id": ticket_id,
        "tenant_id": tenant_id,
        "branch_id": branch_id,
        "user_id": user_id,
        "phone": phone,
        "message": (question or "").strip(),
        "status": "new",
    }
    out_path = Path(PROJECT_ROOT) / "data" / ".cache" / "tickets.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8") if not out_path.exists() else None
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ticket, ensure_ascii=False) + "\n")

    return ToolResult(
        answer="Dạ em đã ghi nhận thông tin. Tư vấn viên sẽ liên hệ với anh/chị sớm nhất ạ.",
        sources=[],
        metadata={"route": "create_ticket", "ticket_created": True, "ticket_id": ticket_id},
    )

