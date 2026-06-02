import re
import unicodedata
from typing import Dict, List, Optional


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


def _fmt_vnd_public(v: int) -> str:
    return f"{int(v):,}".replace(",", ".") + " VND"


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
