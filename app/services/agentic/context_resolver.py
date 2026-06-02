"""
Context-Switch Resolver — handles mid-sentence intent changes and contradictions.

Detects when users change their mind mid-query, e.g.:
- "Tính giá khoá A. À không, khoá B." → override to B
- "Cho em hỏi giá khoá A. Thôi khỏi, em muốn ghi danh." → override intent

Rules:
1. last_intent_wins: parse multi-entity mentions, keep the last one after override cues
2. Detect contradiction markers ("à không", "thôi khỏi", "khoan")
3. If high ambiguity remains → ask 1 confirmation question

Design: pure regex/heuristic — no LLM call, < 1ms.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ResolveResult:
    changed: bool
    original_intent: str
    resolved_intent: Optional[str] = None    # None = keep original
    override_entity: Optional[str] = None    # Replacement entity if detected
    needs_confirmation: bool = False
    confirmation_question: Optional[str] = None
    reason: str = "no_change"


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    t = unicodedata.normalize("NFD", text or "")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = t.replace("đ", "d").replace("Đ", "D").lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


# ---------------------------------------------------------------------------
# Override / contradiction markers
# ---------------------------------------------------------------------------

_OVERRIDE_MARKERS = [
    r"\ba khong\b",
    r"\ba ko\b",
    r"\bthoi khoi\b",
    r"\bkhoan\b.*\b(?:de|cho|doi)\b",
    r"\bkhoan da\b",
    r"\bthoi\b.*\b(?:bo|doi|khong|ko)\b",
    r"\bkhong\b.*\b(?:phai|co|dung)\b.*\b(?:la|ma)\b",
    r"\bma khong\b",
    r"\bthay doi\b",
    r"\bdoi y\b",
    r"\bdoi lai\b",
    r"\bbo qua\b.*\b(?:cau|cai|phan)\b.*\b(?:truoc|tren)\b",
]

# Intent-switch markers: words that suggest a completely different intent
_INTENT_SWITCH_TO_TICKET = [
    r"\bthoi\b.*\b(?:ghi danh|dang ky|tu van vien)\b",
    r"\bghi danh\b",
    r"\bdang ky hoc\b",
    r"\bgap tu van vien\b",
    r"\blien he\b.*\b(?:tu van|nhan vien)\b",
]

_INTENT_SWITCH_TO_COMPARISON = [
    r"\bso sanh\b.*\b(?:voi|va)\b",
    r"\bkhac nhau\b",
]


# ---------------------------------------------------------------------------
# Entity extraction helpers
# ---------------------------------------------------------------------------

def _extract_entity_mentions(text: str) -> List[Tuple[int, str]]:
    """
    Extract quoted or cue-preceded entity names with their position.
    Returns list of (char_position, entity_name).
    """
    entities: List[Tuple[int, str]] = []

    # Quoted entities: "khoá A", 'khoá B'
    for m in re.finditer(r'["\u201c\u201d\'](.*?)["\u201c\u201d\']', text):
        name = m.group(1).strip()
        if name and len(name) >= 2:
            entities.append((m.start(), name))

    # Pattern: "khoá/khóa + NAME" — captures course name candidates
    for m in re.finditer(
        r"(?:khoa|khóa|khoá)\s+([A-ZĐa-zđÀ-ỹ][A-Za-zĐđÀ-ỹ\s\-\.]{1,40})",
        text,
        flags=re.IGNORECASE,
    ):
        name = m.group(1).strip().rstrip(".,;!?")
        if name and len(name) >= 2:
            entities.append((m.start(), name))

    return sorted(entities, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve(
    question: str,
    current_intent: str,
    extracted_args: Optional[Dict[str, Any]] = None,
) -> ResolveResult:
    """
    Analyze the question for mid-sentence intent changes or contradictions.

    Args:
        question: Raw user question
        current_intent: The intent decided by the router
        extracted_args: Arguments already extracted by the argument extractors

    Returns:
        ResolveResult indicating whether intent/entity was overridden
    """
    if not question or not question.strip():
        return ResolveResult(changed=False, original_intent=current_intent, reason="empty")

    qn = _norm(question)
    args = extracted_args or {}

    # --- Step 1: Check for override markers ---
    has_override = False
    override_pos = -1
    for pat in _OVERRIDE_MARKERS:
        m = re.search(pat, qn)
        if m:
            has_override = True
            override_pos = max(override_pos, m.start())
            break

    # --- Step 2: Check for intent switches ---
    # Does the query end with a different intent than what was routed?
    for pat in _INTENT_SWITCH_TO_TICKET:
        m = re.search(pat, qn)
        if m and has_override and m.start() > override_pos:
            return ResolveResult(
                changed=True,
                original_intent=current_intent,
                resolved_intent="create_ticket",
                reason=f"intent_switch:create_ticket after override at {override_pos}",
            )

    for pat in _INTENT_SWITCH_TO_COMPARISON:
        m = re.search(pat, qn)
        if m and has_override and m.start() > override_pos:
            return ResolveResult(
                changed=True,
                original_intent=current_intent,
                resolved_intent="comparison",
                reason=f"intent_switch:comparison after override at {override_pos}",
            )

    # --- Step 3: Entity override (last_intent_wins) ---
    if has_override and current_intent in ("tuition_calculator", "comparison", "course_search"):
        entities = _extract_entity_mentions(question)
        if len(entities) >= 2:
            # Multiple entities + override → take the LAST mentioned entity
            last_entity = entities[-1][1]
            return ResolveResult(
                changed=True,
                original_intent=current_intent,
                override_entity=last_entity,
                reason=f"entity_override:last={last_entity}",
            )

    # --- Step 4: High ambiguity → ask confirmation ---
    # Multiple separate intent cues in the same query without clear override
    intent_cue_count = 0
    if re.search(r"\b(?:hoc phi|gia|bao nhieu tien|phi)\b", qn):
        intent_cue_count += 1
    if re.search(r"\b(?:so sanh|khac nhau)\b", qn):
        intent_cue_count += 1
    if re.search(r"\b(?:ghi danh|dang ky|tu van vien|lien he)\b", qn):
        intent_cue_count += 1

    if intent_cue_count >= 2 and has_override:
        return ResolveResult(
            changed=False,
            original_intent=current_intent,
            needs_confirmation=True,
            confirmation_question=(
                "Dạ em thấy anh/chị vừa đề cập nhiều yêu cầu. "
                "Anh/chị muốn em hỗ trợ việc nào trước ạ?"
            ),
            reason="multi_intent_ambiguity",
        )

    # --- No changes needed ---
    return ResolveResult(changed=False, original_intent=current_intent, reason="no_change")
