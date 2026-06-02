from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional


def parse_money_to_vnd(s: str) -> Optional[int]:
    """
    Parse common VN money notations to integer VND.
    Supports: 9.500.000, 9,500,000, 10tr, 10 triệu, 10.5tr.
    """
    if not s:
        return None
    raw = (s or "").strip().lower()
    raw = raw.replace("₫", "đ").replace("vnđ", "vnd").replace("đồng", "dong")
    # Normalize common symbols to help downstream matching.
    raw = raw.replace("đ", "vnd")

    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(k|nghin|ngan|ngàn|nghìn|canh|cành)\b", raw)
    if m:
        try:
            v = float(m.group(1).replace(",", "."))
            return int(round(v * 1_000))
        except Exception:
            return None

    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(cu|củ)\b", raw)
    if m:
        try:
            v = float(m.group(1).replace(",", "."))
            return int(round(v * 1_000_000))
        except Exception:
            return None

    m = re.search(r"(\d+(?:[.,]\d+)?)\s*tr\b", raw)
    if m:
        v = float(m.group(1).replace(",", "."))
        return int(round(v * 1_000_000))

    m = re.search(r"(\d+(?:[.,]\d+)?)\s*tri[eệ]u", raw)
    if m:
        v = float(m.group(1).replace(",", "."))
        return int(round(v * 1_000_000))

    m = re.search(r"(\d[\d\.\, ]{4,})\s*(vnd|dong)\b", raw)
    if m:
        num = re.sub(r"[^\d]", "", m.group(1))
        if len(num) >= 5:
            return int(num)
    return None


def extract_evidence_dict(text: str) -> Dict[str, object]:
    """
    Structured evidence for filtering: duration + tuition + targets.
    Only returns keys that have values.
    """
    t = text or ""
    evidence: Dict[str, List[object]] = {
        "duration_months": [],
        "duration_weeks": [],
        "tuition_vnd": [],
        "fees_vnd": [],
        "ielts_target": [],
        "toeic_target": [],
        "cefr_target": [],
    }

    def _norm(s: str) -> str:
        s = (s or "").lower()
        try:
            s = unicodedata.normalize("NFD", s)
            s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        except Exception:
            pass
        s = re.sub(r"[^a-z0-9]+", " ", s, flags=re.IGNORECASE).strip()
        return s

    TUITION_HINTS = [
        "hoc phi",
        "tuition",
        "tron goi",
        "niem yet",
        "tong hoc phi",
        "tong chi phi",
        "chi phi",
    ]
    NOT_TUITION_HINTS = [
        "le phi",
        "phi thi",
        "thi thu",
        "phi giu cho",
        "giu cho",
        "phi tai lieu",
        "tai lieu",
        "material",
        "phu thu",
        "phi xep lop",
        "xep lop",
    ]
    TUITION_MIN_VND = 1_000_000

    def _classify_money(vnd: int, around: str) -> str:
        """
        Decide whether a money mention is likely tuition or other fees.
        """
        a = _norm(around)
        has_tuition_hint = any(k in a for k in TUITION_HINTS)
        has_not_tuition_hint = any(k in a for k in NOT_TUITION_HINTS)

        if vnd < TUITION_MIN_VND and not has_tuition_hint:
            return "fee"
        if has_not_tuition_hint and not has_tuition_hint:
            return "fee"
        return "tuition"

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

    for m in re.finditer(r"(?i)(?:^|\s)(\d[\d\., ]{4,}\s*(?:₫|đ|vnd|dong)\b)", t):
        v = parse_money_to_vnd(m.group(1))
        if v is not None:
            around = t[max(0, m.start(1) - 60) : min(len(t), m.end(1) + 60)]
            if _classify_money(int(v), around) == "tuition":
                evidence["tuition_vnd"].append(int(v))
            else:
                evidence["fees_vnd"].append(int(v))
    for m in re.finditer(r"(?i)\b(\d+(?:[.,]\d+)?)\s*(tr|tri[eệ]u)\b", t):
        v = parse_money_to_vnd(m.group(0))
        if v is not None:
            around = t[max(0, m.start(0) - 60) : min(len(t), m.end(0) + 60)]
            if _classify_money(int(v), around) == "tuition":
                evidence["tuition_vnd"].append(int(v))
            else:
                evidence["fees_vnd"].append(int(v))

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

    def uniq_sorted(vals: List[object]) -> List[object]:
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
        evidence[k] = uniq_sorted(evidence[k])

    return {k: v for k, v in evidence.items() if v}
