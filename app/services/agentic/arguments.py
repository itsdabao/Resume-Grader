from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from .evidence import parse_money_to_vnd
from .preprocess import extract_phone


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D").replace("Ž`", "d").replace("Ž?", "D")


def _norm_ascii(s: str) -> str:
    s = _strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def _parse_percent(text: str) -> Optional[float]:
    m = re.search(r"(?i)(\d{1,3}(?:[.,]\d+)?)\s*%", text or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except Exception:
        return None


def _iter_money_mentions(text: str) -> List[Tuple[int, int, int, str]]:
    """
    Return list of (vnd, start, end, raw_fragment).
    Includes common VND notations and slang (via parse_money_to_vnd).
    """
    out: List[Tuple[int, int, int, str]] = []
    t = text or ""
    patterns = [
        r"(?i)\b\d[\d\., ]{0,16}\s*(?:vnd|dong|đ)\b",
        r"(?i)\b\d+(?:[.,]\d+)?\s*(?:tr|tri[eê]u)\b",
        r"(?i)\b\d+(?:[.,]\d+)?\s*(?:k|nghin|ngan|ngàn|nghìn)\b",
        r"(?i)\b\d+(?:[.,]\d+)?\s*(?:cu|củ)\b",
        r"(?i)\b\d+(?:[.,]\d+)?\s*(?:canh|cành)\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, t):
            raw = m.group(0)
            v = parse_money_to_vnd(raw)
            if v is not None and v > 0:
                out.append((int(v), m.start(0), m.end(0), raw))
    return out


def _discount_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"(?i)\b(?:giam|giảm)\s+([^\n,;]{1,30})", text or ""):
        frag = m.group(1) or ""
        if "%" in frag:
            continue
        spans.append((m.start(1), m.end(1)))
    return spans


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


DiscountType = Literal["percent", "cash"]
DiscountScope = Literal["tuition", "total"]
DiscountPeriod = Literal["one_time", "per_month"]


def extract_tuition_calculator_args(question: str) -> Dict[str, object]:
    q = (question or "").strip()
    qn = _norm_ascii(q)

    discount_percent = _parse_percent(q)
    discount_type: Optional[DiscountType] = "percent" if discount_percent is not None else None
    discount_value: Optional[object] = discount_percent

    discount_amount: Optional[int] = None
    for m in re.finditer(r"(?i)\b(?:giam|giảm)\s+([^\n,;]{1,30})", q):
        frag = (m.group(1) or "").strip()
        if not frag or "%" in frag:
            continue
        v = parse_money_to_vnd(frag)
        if isinstance(v, int) and v > 0:
            discount_amount = int(v)
            break
    if discount_amount is not None and discount_percent is None:
        discount_type = "cash"
        discount_value = int(discount_amount)

    scope: DiscountScope = "tuition"
    if any(
        k in qn
        for k in [
            "giam tong",
            "giam tren tong",
            "tren tong",
            "tong chi phi",
            "tong tien",
            "tong thanh toan",
            "tong cong",
        ]
    ):
        scope = "total"

    period: DiscountPeriod = "one_time"
    if any(k in qn for k in ["moi thang", "tung thang", "hang thang", "/thang", "per month"]):
        period = "per_month"

    fee_hints = [
        "giao trinh",
        "tai lieu",
        "hoc lieu",
        "material",
        "gui xe",
        "phu phi",
        "le phi",
        "phi xep lop",
        "phi hoc lieu",
        "xep lop",
    ]
    tuition_hints = ["hoc phi", "goi hoc", "tron goi", "tuition"]

    spans = _discount_spans(q)
    mentions = _iter_money_mentions(q)

    base_price: Optional[int] = None
    base_candidates: List[int] = []
    surcharges: List[Dict[str, object]] = []
    for v, s, e, raw in mentions:
        around = _norm_ascii((q or "")[max(0, s - 40) : min(len(q), e + 40)])

        if discount_amount is not None and v == int(discount_amount) and any(_overlaps((s, e), sp) for sp in spans):
            continue
        if any(_overlaps((s, e), sp) for sp in spans):
            continue

        if v >= 1_000_000 and any(h in around for h in tuition_hints):
            base_price = int(v) if base_price is None else max(int(base_price), int(v))
            continue

        if any(h in around for h in fee_hints):
            surcharges.append({"amount": int(v), "label": raw.strip()})
            continue

        base_candidates.append(int(v))

    if base_price is None and base_candidates:
        base_price = max(base_candidates)

    return {
        "base_price": base_price,
        "discount_value": discount_value,
        "discount_type": discount_type,
        "discount_scope": scope,
        "discount_period": period,
        "surcharge": surcharges,
    }


Criteria = Literal["price", "duration", "output", "all"]


def extract_comparison_args(question: str) -> Dict[str, object]:
    q = (question or "").strip()
    qn = _norm_ascii(q)

    crit_hits: List[Criteria] = []
    if any(k in qn for k in ["hoc phi", "gia", "price", "phi"]):
        crit_hits.append("price")
    if any(k in qn for k in ["thoi luong", "bao lau", "so thang", "tuan", "buoi", "duration"]):
        crit_hits.append("duration")
    if any(k in qn for k in ["dau ra", "output", "muc tieu", "band", "diem"]):
        crit_hits.append("output")
    if any(k in qn for k in ["tat ca", "toan bo", "full", "all"]):
        crit_hits = []

    criteria: Criteria = crit_hits[0] if len(set(crit_hits)) == 1 else "all"

    t = q
    t = re.sub(r"(?i)\b(vs|voi|với)\b", " vs ", t)
    t = re.sub(r"(?i)\b(hay|hoac|hoặc)\b", " vs ", t)
    m = re.search(r"(?i)\bso\s*s[aá]nh\b(.*)$", t)
    if m:
        t = (m.group(1) or "").strip()

    parts = [p.strip() for p in t.split(" vs ") if p.strip()]
    subjects: List[str] = []
    for p in parts[:4]:
        p = re.sub(r"(?i)\b(giua|khoa hoc|khóa học|lop|lớp|course|nen hoc|nên học)\b", " ", p).strip()
        p = re.split(r"[:;,.\n\-]", p, maxsplit=1)[0].strip()
        if p and p not in subjects:
            subjects.append(p[:80])

    return {"subjects": subjects, "criteria": criteria}


@dataclass(frozen=True)
class TicketArgs:
    phone: Optional[str]
    name: Optional[str]
    preferred_time: Optional[str]
    intent_summary: str


def _extract_name(text: str) -> Optional[str]:
    t = (text or "").strip()
    m = re.search(r"(?i)\b(?:tôi|toi|em|mình|minh|anh|chị|chi)\s+(?:là|la|tên|ten)\s+([^\n,;.]{2,40})", t)
    if not m:
        m = re.search(r"(?i)\b(?:toi|em|minh|anh|chi)\s+(?:la|ten)\s+([^\n,;.]{2,40})", _strip_accents(t))
        if not m:
            return None
    name = re.sub(r"\s+", " ", (m.group(1) or "").strip())
    name = re.sub(r"(?i)\b(sdt|phone|so dien thoai).*$", "", name).strip()
    if not name:
        return None
    if len(name) > 40:
        name = name[:40].strip()
    return name


def _extract_preferred_time(text: str) -> Optional[str]:
    t = (text or "").strip()
    tn = _norm_ascii(t)

    for phrase in ["hom nay", "ngay mai", "mai", "thu 2", "thu 3", "thu 4", "thu 5", "thu 6", "thu 7", "chu nhat", "sang", "chieu", "toi"]:
        if phrase in tn:
            return phrase
    m = re.search(r"(?i)\b(\d{1,2})(?:[:hH](\d{2}))\b", t)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}"
    m = re.search(r"(?i)\b(\d{1,2})\s*h\b", t)
    if m:
        hh = int(m.group(1))
        if 0 <= hh <= 23:
            return f"{hh:02d}:00"
    return None


def extract_ticket_args(question: str) -> Dict[str, object]:
    q = (question or "").strip()
    phone = extract_phone(q)
    name = _extract_name(q)
    preferred_time = _extract_preferred_time(q)

    intent = q
    if phone:
        intent = re.sub(re.escape(phone), "", intent)
    intent = re.sub(r"\s+", " ", intent).strip()
    if len(intent) > 180:
        intent = intent[:180].strip()

    return {
        "phone": phone,
        "name": name,
        "preferred_time": preferred_time,
        "intent_summary": intent,
    }

