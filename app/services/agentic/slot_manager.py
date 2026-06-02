"""
Slot Manager — unified slot-filling system for the agentic pipeline.

Each intent has required and optional slots. After routing determines the intent,
the slot manager checks whether extracted arguments satisfy the required slots.
If not, it generates a natural Vietnamese clarification question.

Design: pure deterministic — no LLM call, < 1ms.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SlotResult:
    complete: bool
    missing: List[str]
    clarify_question: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Intent slot definitions
# ---------------------------------------------------------------------------

INTENT_SLOTS: Dict[str, Dict[str, Any]] = {
    "tuition_calculator": {
        "required": ["course_name"],
        "optional": ["discount_type", "group_size", "payment_method", "voucher_code"],
        "clarify_templates": {
            "course_name": (
                "Dạ anh/chị cho em biết **tên khoá học** cần tính phí nhé. "
                "Ví dụ: IELTS Fundamentals, Fluent Speaking, Corporate English, ..."
            ),
        },
    },
    "comparison": {
        "required": ["entity_a", "entity_b"],
        "optional": ["criteria"],
        "clarify_templates": {
            "entity_a": (
                "Dạ anh/chị muốn so sánh những khoá học nào ạ? "
                "Vui lòng cho em biết ít nhất **2 khoá học** cần so sánh nhé."
            ),
            "entity_b": (
                "Dạ em thấy anh/chị nhắc đến 1 khoá. Anh/chị muốn so sánh với **khoá nào** nữa ạ?"
            ),
        },
    },
    "create_ticket": {
        "required": ["phone"],
        "optional": ["name", "preferred_time", "course_interest"],
        "clarify_templates": {
            "phone": (
                "Dạ anh/chị cho em xin **số điện thoại** để tư vấn viên liên hệ hỗ trợ chi tiết nhé. "
                "Anh/chị cũng có thể cho em biết **khung giờ thuận tiện** để được gọi lại ạ."
            ),
        },
    },
    "course_search": {
        "required": [],  # Free-form queries are OK
        "optional": ["level", "goal", "budget", "schedule", "target_score"],
        "clarify_templates": {},
    },
}


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    t = unicodedata.normalize("NFD", text or "")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = t.replace("đ", "d").replace("Đ", "D").lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


# ---------------------------------------------------------------------------
# Sanity / reasonableness checks
# ---------------------------------------------------------------------------

_UNREALISTIC_BUDGET_VND = 500_000  # < 500k VND is unrealistically low for a course
_UNREALISTIC_TIMELINE_DAYS = 7     # < 7 days to master something


def _check_reasonableness(
    intent: str,
    extracted_args: Dict[str, Any],
    question: str,
) -> List[str]:
    """Return list of warning strings for unrealistic expectations."""
    warnings: List[str] = []
    qn = _norm(question)

    # Budget too low
    budget = extracted_args.get("budget_vnd") or extracted_args.get("budget")
    if budget is not None:
        try:
            budget_val = int(budget)
            if 0 < budget_val < _UNREALISTIC_BUDGET_VND:
                warnings.append(
                    f"Dạ mức ngân sách {budget_val:,} VND có thể chưa đủ cho một khoá học đầy đủ. "
                    "Em có thể tư vấn các lựa chọn phù hợp nhất với ngân sách của anh/chị ạ."
                )
        except (ValueError, TypeError):
            pass

    # Unrealistic timeline
    timeline_cues = ["1 tuan", "1 ngay", "2 ngay", "3 ngay", "vai ngay", "may ngay"]
    mastery_cues = ["thanh thao", "gioi", "fluent", "ielts 7", "ielts 8", "ielts 9"]
    has_short_timeline = any(c in qn for c in timeline_cues)
    has_mastery_goal = any(c in qn for c in mastery_cues)
    if has_short_timeline and has_mastery_goal:
        warnings.append(
            "Dạ mục tiêu này thường cần thời gian luyện tập lâu hơn để đạt hiệu quả tốt nhất. "
            "Em có thể tư vấn lộ trình phù hợp và thực tế cho anh/chị ạ."
        )

    # Age check (if mentioned in query, not extracted args — crude heuristic)
    age_match = re.search(r"\b(\d{1,2})\s*tuoi\b", qn)
    if age_match:
        age = int(age_match.group(1))
        if age < 5:
            warnings.append(
                f"Dạ bé {age} tuổi có thể chưa phù hợp với các khoá học hiện tại. "
                "Em sẽ kiểm tra xem trung tâm có chương trình nào phù hợp cho bé nhé."
            )

    return warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_slots(
    intent: str,
    extracted_args: Dict[str, Any],
    question: str = "",
) -> SlotResult:
    """
    Check if extracted arguments satisfy the required slots for the given intent.

    Returns SlotResult with:
    - complete=True if all required slots are filled
    - missing: list of missing slot names
    - clarify_question: natural Vietnamese question asking for missing info
    - warnings: list of reasonableness warnings (unrealistic budget, timeline, etc.)
    """
    config = INTENT_SLOTS.get(intent)
    if config is None:
        # Unknown intent — pass through (don't block)
        return SlotResult(complete=True, missing=[])

    required = config.get("required", [])
    templates = config.get("clarify_templates", {})

    missing: List[str] = []
    for slot_name in required:
        value = extracted_args.get(slot_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(slot_name)

    # Reasonableness checks
    warnings = _check_reasonableness(intent, extracted_args, question)

    if not missing:
        return SlotResult(complete=True, missing=[], warnings=warnings)

    # Build clarification question from the FIRST missing required slot
    first_missing = missing[0]
    clarify = templates.get(first_missing)
    if clarify is None:
        # Generic fallback
        slot_vi = {
            "course_name": "tên khoá học",
            "entity_a": "khoá học thứ nhất",
            "entity_b": "khoá học thứ hai",
            "phone": "số điện thoại",
            "name": "tên",
        }
        label = slot_vi.get(first_missing, first_missing)
        clarify = f"Dạ anh/chị cho em biết thêm **{label}** để em hỗ trợ chính xác nhé."

    return SlotResult(
        complete=False,
        missing=missing,
        clarify_question=clarify,
        warnings=warnings,
    )
