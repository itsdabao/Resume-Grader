from __future__ import annotations

import unicodedata
import re
from dataclasses import dataclass
from typing import Optional


VN_PHONE_RE = re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s\.-]?\d){8,10}(?!\d)")
VN_PHONE_MOBILE_RE = re.compile(r"^0(3|5|7|8|9)\d{8}$")  # 10 digits
VN_PHONE_LANDLINE_RE = re.compile(r"^02\d{9}$")  # 11 digits (02 + 9 digits)


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    # Vietnamese "đ/Đ" does not decompose under NFD.
    return s.replace("đ", "d").replace("Đ", "D")


def detect_language(text: str) -> str:
    """
    Cheap VI/EN detector (heuristic; no external deps).
    """
    s = (text or "").strip()
    if not s:
        return "vi"
    # Vietnamese diacritics check
    if re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", s.lower()):
        return "vi"

    s_norm = _strip_accents(s).lower()
    # Very short English queries should still be detected as EN (e.g., "Price?")
    if len(s_norm) <= 12 and re.fullmatch(r"[a-z0-9\?\!\.\s]+", s_norm or ""):
        if re.search(r"\b(price|tuition|schedule|fee|where|when|how|what)\b", s_norm):
            return "en"

    # Strong Vietnamese keyword hints even without accents
    if re.search(r"\b(hoc phi|lich hoc|khai giang|uu dai|tu van|dang ky|co so|chi nhanh)\b", s_norm):
        return "vi"

    en_hits = len(re.findall(r"\b(the|a|an|is|are|do|does|how|what|where|when|price|tuition|schedule|fee)\b", s_norm))
    vi_hits = len(re.findall(r"\b(hoc|phi|lich|khai|giang|uu|dai|tu|van|dang|ky)\b", s_norm))
    if en_hits > vi_hits and en_hits >= 1:
        return "en"
    return "vi"


def extract_phone(text: str) -> Optional[str]:
    m = VN_PHONE_RE.search(text or "")
    if not m:
        return None
    raw = m.group(0)
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("84"):
        digits = "0" + digits[2:]
    # Validate VN phone shapes to avoid catching IDs/accounts.
    if not (VN_PHONE_MOBILE_RE.match(digits) or VN_PHONE_LANDLINE_RE.match(digits)):
        return None
    return digits


TOXIC_KEYWORDS = [
    # Keep short + conservative to avoid false positives.
    "địt",
    "dit",
    "đụ",
    "đụ",
    "lồn",
    "lon ",
    "cặc",
    "cak",
    "đm",
    "dm ",
    "vcl",
]


def _normalize_for_toxic(text: str) -> tuple[list[str], str]:
    """
    Normalize variants like "v.c.l", "d_m" by:
    - stripping accents
    - lowercasing
    - tokenizing on non-alnum
    - also building a compact string with separators removed
    """
    s = _strip_accents(text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", s)
    compact = "".join(tokens)
    return tokens, compact


def is_toxic(text: str) -> bool:
    tokens, compact = _normalize_for_toxic(text or "")
    token_set = set(tokens)

    # Exact token matches
    # NOTE: Avoid common Vietnamese words that become ambiguous after accent-stripping.
    # - "các" -> "cac" (very common, not toxic)
    # - "du" appears in many benign words ("du học", "đủ", ...)
    if any(k in token_set for k in ["vcl", "dm", "dmm", "dit"]):
        return True

    # Compact matches (handles punctuation-separated variants like "d.m", "v.c.l")
    # Only apply if tokens are mostly single characters to avoid merging normal words
    # (e.g. "bad man" -> "dm" should NOT trigger).
    if len(tokens) >= 2 and all(len(t) == 1 for t in tokens):
        if any(k in compact for k in ["vcl", "dm", "dmm", "ditme"]):
            return True

    # Backward compatibility list (substring on original lower)
    # Backward compatibility list (substring on original lower)
    t = (text or "").lower()
    return any(k in t for k in TOXIC_KEYWORDS)




@dataclass(frozen=True)
class PreprocessResult:
    query: str
    language: str
    phone: Optional[str]
    toxic: bool


def preprocess_query(query: str) -> PreprocessResult:
    q = (query or "").strip()
    return PreprocessResult(
        query=q,
        language=detect_language(q),
        phone=extract_phone(q),
        toxic=is_toxic(q),
    )
