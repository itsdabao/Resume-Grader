"""
Policy Registry — JSON-based rule engine for tenant-specific business rules.

Each tenant has a policy file at ``data/policies/{tenant_id}.json``.
If no file exists, ``_DEFAULT_POLICY`` is used as fallback.

Covers:
- Discount stacking rules (highest_only / combinable / non_stackable)
- Refund eligibility
- Bảo lưu (course hold) rules
- Chuyển nhượng (transfer) rules
- Tuyển sinh eligibility (age limits)
- Combo discount rules
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

_POLICIES_DIR = Path(PROJECT_ROOT) / "data" / "policies"

# ---------------------------------------------------------------------------
# Default policy (used when tenant has no file)
# ---------------------------------------------------------------------------

_DEFAULT_POLICY: Dict[str, Any] = {
    "discount": {
        "stacking": "highest_only",
        "max_percent": 30,
        "combo_allowed": True,
        "flash_sale_combinable": False,
        "voucher_combinable_with_percent": False,
    },
    "refund": {
        "allowed": True,
        "within_sessions": 3,
        "refund_percent": 70,
        "requires_docs": False,
        "note": "Hoàn phí 70% nếu rút trước buổi thứ 3, không hoàn sau buổi 3.",
    },
    "bao_luu": {
        "allowed": True,
        "max_months": 3,
        "requires_medical_cert": False,
        "fee_vnd": 0,
        "note": "Bảo lưu miễn phí tối đa 3 tháng, cần nộp đơn trước 7 ngày.",
    },
    "transfer": {
        "allowed": True,
        "fee_vnd": 200_000,
        "family_only": True,
        "note": "Chuyển nhượng cho người thân trực hệ, phí chuyển 200.000 VND.",
    },
    "class_change": {
        "allowed": True,
        "fee_vnd": 0,
        "same_course_only": True,
        "note": "Đổi lịch học miễn phí trong cùng khoá, tuỳ slot trống.",
    },
    "eligibility": {
        "min_age": None,
        "max_age": None,
        "note": None,
    },
    "commitment": {
        "has_output_guarantee": False,
        "retry_free": False,
        "note": None,
    },
}

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_POLICY_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_from_disk(tenant_id: str) -> Dict[str, Any]:
    """Load policy JSON from disk, return default if missing."""
    tid = str(tenant_id or "").strip().lower()
    if not tid:
        return dict(_DEFAULT_POLICY)

    path = _POLICIES_DIR / f"{tid}.json"
    if not path.exists():
        # Try _default.json as shared fallback
        default_path = _POLICIES_DIR / "_default.json"
        if default_path.exists():
            path = default_path
        else:
            return dict(_DEFAULT_POLICY)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("policy_registry: invalid format in %s, using default", path)
            return dict(_DEFAULT_POLICY)
        # Merge with default so missing keys are filled
        merged = dict(_DEFAULT_POLICY)
        for section_key, section_val in data.items():
            if isinstance(section_val, dict) and isinstance(merged.get(section_key), dict):
                merged[section_key] = {**merged[section_key], **section_val}
            else:
                merged[section_key] = section_val
        return merged
    except Exception as e:
        logger.warning("policy_registry: failed to load %s: %s", path, e)
        return dict(_DEFAULT_POLICY)


def load_policy(tenant_id: str) -> Dict[str, Any]:
    """Load tenant policy with in-memory cache."""
    tid = str(tenant_id or "").strip().lower()
    if tid in _POLICY_CACHE:
        return _POLICY_CACHE[tid]
    policy = _load_from_disk(tid)
    _POLICY_CACHE[tid] = policy
    return policy


def clear_cache(tenant_id: Optional[str] = None) -> None:
    """Clear policy cache. Call after re-ingesting or updating policy files."""
    if tenant_id:
        _POLICY_CACHE.pop(str(tenant_id).strip().lower(), None)
    else:
        _POLICY_CACHE.clear()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscountResult:
    applied_percent: float
    applied_amount_vnd: int
    stacking_rule: str
    note: str


@dataclass(frozen=True)
class PolicyAnswer:
    """Generic answer from policy lookup."""
    allowed: bool
    note: str
    details: Dict[str, Any]


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------

def has_non_stack_rule(tenant_id: str) -> bool:
    """Drop-in replacement for the old ``_tenant_policy_has_non_stack_rule``."""
    policy = load_policy(tenant_id)
    discount = policy.get("discount", {})
    stacking = str(discount.get("stacking", "highest_only")).lower()
    return stacking in ("highest_only", "non_stackable")


def resolve_discount(
    tenant_id: str,
    *,
    discounts: List[Dict[str, Any]],
) -> DiscountResult:
    """
    Given a list of candidate discounts, apply tenant stacking rules.

    Each discount dict: {"type": str, "percent": float|None, "amount_vnd": int|None}
    """
    policy = load_policy(tenant_id)
    discount_cfg = policy.get("discount", {})
    stacking = str(discount_cfg.get("stacking", "highest_only")).lower()
    max_pct = float(discount_cfg.get("max_percent", 100))

    if not discounts:
        return DiscountResult(0.0, 0, stacking, "Không có giảm giá.")

    percents = [float(d.get("percent") or 0) for d in discounts if d.get("percent")]
    amounts = [int(d.get("amount_vnd") or 0) for d in discounts if d.get("amount_vnd")]

    if stacking in ("highest_only", "non_stackable"):
        best_pct = max(percents) if percents else 0.0
        best_amt = max(amounts) if amounts else 0
        best_pct = min(best_pct, max_pct)
        note = "Áp dụng chính sách KHÔNG CỘNG DỒN — chỉ lấy ưu đãi cao nhất."
        return DiscountResult(best_pct, best_amt, stacking, note)
    elif stacking == "combinable":
        total_pct = min(sum(percents), max_pct)
        total_amt = sum(amounts)
        note = "Áp dụng chính sách CỘNG DỒN các ưu đãi."
        return DiscountResult(total_pct, total_amt, stacking, note)
    else:
        best_pct = max(percents) if percents else 0.0
        best_amt = max(amounts) if amounts else 0
        return DiscountResult(min(best_pct, max_pct), best_amt, stacking, "Chính sách mặc định: lấy cao nhất.")


def check_refund(tenant_id: str, *, sessions_attended: int = 0) -> PolicyAnswer:
    """Check refund eligibility."""
    policy = load_policy(tenant_id)
    refund = policy.get("refund", {})
    allowed = bool(refund.get("allowed", False))
    max_sessions = int(refund.get("within_sessions", 0))
    pct = int(refund.get("refund_percent", 0))
    note = str(refund.get("note") or "Không có thông tin chính sách hoàn phí.")

    if not allowed:
        return PolicyAnswer(False, "Trung tâm không có chính sách hoàn phí.", {"refund": refund})

    eligible = sessions_attended <= max_sessions
    if eligible:
        detail_note = f"Hoàn {pct}% học phí (đã học {sessions_attended}/{max_sessions} buổi). {note}"
    else:
        detail_note = f"Không đủ điều kiện hoàn phí (đã học {sessions_attended} buổi, giới hạn {max_sessions}). {note}"

    return PolicyAnswer(eligible, detail_note, {"refund": refund, "sessions_attended": sessions_attended})


def check_bao_luu(tenant_id: str) -> PolicyAnswer:
    """Check course hold (bảo lưu) policy."""
    policy = load_policy(tenant_id)
    bl = policy.get("bao_luu", {})
    allowed = bool(bl.get("allowed", False))
    note = str(bl.get("note") or "Không có thông tin chính sách bảo lưu.")
    return PolicyAnswer(allowed, note, {"bao_luu": bl})


def check_transfer(tenant_id: str) -> PolicyAnswer:
    """Check transfer (chuyển nhượng) policy."""
    policy = load_policy(tenant_id)
    tr = policy.get("transfer", {})
    allowed = bool(tr.get("allowed", False))
    note = str(tr.get("note") or "Không có thông tin chính sách chuyển nhượng.")
    return PolicyAnswer(allowed, note, {"transfer": tr})


def check_class_change(tenant_id: str) -> PolicyAnswer:
    """Check class schedule change policy."""
    policy = load_policy(tenant_id)
    cc = policy.get("class_change", {})
    allowed = bool(cc.get("allowed", False))
    note = str(cc.get("note") or "Không có thông tin chính sách đổi lịch.")
    return PolicyAnswer(allowed, note, {"class_change": cc})


def check_eligibility(tenant_id: str, *, age: Optional[int] = None) -> PolicyAnswer:
    """Check student eligibility (age, prerequisites, etc)."""
    policy = load_policy(tenant_id)
    elig = policy.get("eligibility", {})
    min_age = elig.get("min_age")
    max_age = elig.get("max_age")
    note = str(elig.get("note") or "")

    if age is not None:
        if min_age is not None and age < int(min_age):
            return PolicyAnswer(
                False,
                f"Rất tiếc, trung tâm chỉ nhận học viên từ {min_age} tuổi trở lên. {note}".strip(),
                {"eligibility": elig, "student_age": age},
            )
        if max_age is not None and age > int(max_age):
            return PolicyAnswer(
                False,
                f"Rất tiếc, trung tâm chỉ nhận học viên đến {max_age} tuổi. {note}".strip(),
                {"eligibility": elig, "student_age": age},
            )

    return PolicyAnswer(True, note or "Đủ điều kiện tuyển sinh.", {"eligibility": elig})


def check_commitment(tenant_id: str) -> PolicyAnswer:
    """Check output guarantee / commitment policy."""
    policy = load_policy(tenant_id)
    cm = policy.get("commitment", {})
    has_guarantee = bool(cm.get("has_output_guarantee", False))
    note = str(cm.get("note") or "Không có thông tin cam kết đầu ra.")
    return PolicyAnswer(has_guarantee, note, {"commitment": cm})
