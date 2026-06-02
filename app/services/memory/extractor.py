from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Tuple

from app.core.config import MEMORY_HEURISTIC_REGEX_CONFIDENCE, MEMORY_HEURISTIC_TOOL_CONFIDENCE
from app.services.agentic.evidence import parse_money_to_vnd


def _norm_text(s: str) -> str:
    t = unicodedata.normalize("NFD", s or "")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = t.replace("đ", "d").replace("Đ", "D")
    return t.lower().strip()


def _add_fact_unique(
    out: List[Dict[str, Any]],
    seen: set[tuple[str, str]],
    *,
    fact_key: str,
    fact_value_text: str,
    confidence: float,
    source: str,
    ttl_days: int,
    fact_value_json: Dict[str, Any] | None = None,
) -> None:
    key = str(fact_key or "").strip()
    val = str(fact_value_text or "").strip()
    if not key or not val:
        return
    dedupe = (key, _norm_text(val))
    if dedupe in seen:
        return
    seen.add(dedupe)
    out.append(
        {
            "fact_key": key,
            "fact_value_text": val,
            "fact_value_json": (fact_value_json or {}) if isinstance(fact_value_json or {}, dict) else {},
            "confidence": float(confidence),
            "source": str(source or "heuristic"),
            "ttl_days": int(ttl_days),
        }
    )


def _extract_from_tool_metadata(
    *,
    tool_metadata: Dict[str, Any] | None,
    route: str | None,
    profile_patch: Dict[str, Any],
    facts: List[Dict[str, Any]],
    seen: set[tuple[str, str]],
) -> None:
    md = tool_metadata if isinstance(tool_metadata, dict) else {}
    args = md.get("extracted_args") if isinstance(md.get("extracted_args"), dict) else {}
    if not args:
        return

    conf = float(MEMORY_HEURISTIC_TOOL_CONFIDENCE)
    src = "heuristic_tool"

    phone = str(args.get("phone") or "").strip()
    if phone:
        profile_patch["phone"] = phone
        _add_fact_unique(
            facts,
            seen,
            fact_key="contact_phone",
            fact_value_text=phone,
            confidence=conf,
            source=src,
            ttl_days=365,
        )

    course = str(
        args.get("course_name")
        or args.get("course")
        or args.get("program")
        or args.get("course_interest")
        or ""
    ).strip()
    if course:
        _add_fact_unique(
            facts,
            seen,
            fact_key="course_interest",
            fact_value_text=course,
            confidence=conf,
            source=src,
            ttl_days=180,
        )

    target = str(
        args.get("target_level")
        or args.get("target_band")
        or args.get("target_score")
        or args.get("goal")
        or ""
    ).strip()
    if target:
        _add_fact_unique(
            facts,
            seen,
            fact_key="target_level",
            fact_value_text=target,
            confidence=conf,
            source=src,
            ttl_days=365,
        )

    if str(route or "").strip().lower() == "tuition_calculator":
        if isinstance(md.get("computed_final_vnd"), (int, float)):
            val = int(md["computed_final_vnd"])
            _add_fact_unique(
                facts,
                seen,
                fact_key="latest_quote_vnd",
                fact_value_text=f"{val} VND",
                fact_value_json={"value_vnd": val},
                confidence=conf,
                source=src,
                ttl_days=30,
            )


def _extract_from_user_text(
    *,
    user_messages: List[str],
    facts: List[Dict[str, Any]],
    seen: set[tuple[str, str]],
) -> None:
    conf = float(MEMORY_HEURISTIC_REGEX_CONFIDENCE)
    src = "heuristic_regex"

    intent_cues = (
        "muc tieu",
        "target",
        "can dat",
        "muon dat",
        "phai dat",
        "goal",
        "band",
    )
    budget_cues = ("ngan sach", "budget", "toi da", "toi co", "tam")

    for content in user_messages:
        txt = str(content or "").strip()
        if not txt:
            continue
        norm = _norm_text(txt)

        ielts_match = re.search(r"\bielts\s*(\d(?:\.\d)?)\b", norm, flags=re.IGNORECASE)
        if ielts_match and any(cue in norm for cue in intent_cues):
            score = ielts_match.group(1)
            _add_fact_unique(
                facts,
                seen,
                fact_key="mentioned_ielts_target",
                fact_value_text=f"IELTS {score}",
                confidence=conf,
                source=src,
                ttl_days=30,
            )

        grade_match = re.search(r"\b(?:dang hoc|hoc lop|lop)\s*(\d{1,2})\b", norm, flags=re.IGNORECASE)
        if grade_match:
            grade = grade_match.group(1)
            _add_fact_unique(
                facts,
                seen,
                fact_key="mentioned_grade",
                fact_value_text=f"Lop {grade}",
                confidence=conf,
                source=src,
                ttl_days=180,
            )

        if any(c in norm for c in budget_cues):
            for m in re.finditer(
                r"(\d[\d\.\,\s]{0,12}(?:\s*(?:tr|trieu|k|vnd|đ))?)",
                txt,
                flags=re.IGNORECASE,
            ):
                frag = str(m.group(1) or "").strip()
                if not frag:
                    continue
                vnd = parse_money_to_vnd(frag)
                if vnd is None:
                    continue
                ivnd = int(vnd)
                if ivnd <= 0:
                    continue
                _add_fact_unique(
                    facts,
                    seen,
                    fact_key="mentioned_budget_vnd",
                    fact_value_text=f"{ivnd} VND",
                    fact_value_json={"value_vnd": ivnd},
                    confidence=conf,
                    source=src,
                    ttl_days=30,
                )
                break


def heuristic_extract(
    *,
    tenant_id: str,
    user_id: str,
    recent_messages: List[Dict[str, Any]],
    tool_metadata: Dict[str, Any] | None = None,
    route: str | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Hook 2A extractor:
    - Tool-derived facts: high confidence
    - Regex-derived facts: low confidence
    """
    _ = tenant_id, user_id  # reserved for future rulesets
    profile_patch: Dict[str, Any] = {}
    facts: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    _extract_from_tool_metadata(
        tool_metadata=tool_metadata,
        route=route,
        profile_patch=profile_patch,
        facts=facts,
        seen=seen,
    )

    user_messages = [
        str(m.get("content") or "")
        for m in (recent_messages or [])
        if isinstance(m, dict) and str(m.get("role") or "").lower() == "user"
    ]
    if user_messages:
        _extract_from_user_text(user_messages=user_messages, facts=facts, seen=seen)

    return profile_patch, facts

