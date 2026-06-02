from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from llama_index.core import VectorStoreIndex

from app.core.config import (
    DATABASE_URL,
    MEMORY_ENABLED,
    MEMORY_LONGTERM_MAX_FACTS,
    MEMORY_LONGTERM_MIN_CONFIDENCE,
)
from app.services.agentic.service import agentic_query

from .background import submit_heuristic_extraction
from .manager import build_history_from_session, maybe_rollup_summary, update_session_after_turn
from .store import get_customer_profile, get_or_create_session, list_memory_facts, upsert_customer_profile


logger = logging.getLogger(__name__)


def build_session_id(*, tenant_id: str, channel: str, user_id: str) -> str:
    """
    SaaS-safe session key: {tenant}:{channel}:{user_id}
    """
    return f"{tenant_id}:{channel}:{user_id}"


def _to_short_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts: List[str] = []
        for k, v in value.items():
            if v is None:
                continue
            parts.append(f"{k}={v}")
            if len(parts) >= 6:
                break
        return ", ".join(parts)
    if isinstance(value, list):
        return ", ".join([str(x) for x in value[:6]])
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _format_long_term_memory_block(
    *,
    profile: Optional[Dict[str, Any]],
    facts: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("[Long-term Memory]")
    lines.append("<user_context>")

    profile_json = {}
    if isinstance(profile, dict):
        maybe_profile = profile.get("profile_json")
        if isinstance(maybe_profile, dict):
            profile_json = maybe_profile
    if profile_json:
        lines.append("[Customer Profile]")
        for k, v in profile_json.items():
            if str(k).startswith("_"):
                continue
            vv = _to_short_text(v)
            if vv:
                lines.append(f"- {k}: {vv}")

    active_facts = [f for f in facts if isinstance(f, dict)]
    if active_facts:
        if profile_json:
            lines.append("")
        lines.append("[Known Facts]")
        for fact in active_facts[:8]:
            key = str(fact.get("fact_key") or "").strip()
            text_val = str(fact.get("fact_value_text") or "").strip()
            json_val = fact.get("fact_value_json") if isinstance(fact.get("fact_value_json"), dict) else {}
            conf = fact.get("confidence")

            if not key:
                continue
            val = text_val or _to_short_text(json_val)
            if not val:
                continue
            if isinstance(conf, (int, float)):
                lines.append(f"- {key}: {val} (confidence: {float(conf):.2f})")
            else:
                lines.append(f"- {key}: {val}")

    lines.append("</user_context>")
    return "\n".join(lines)


def _build_long_term_history(*, tenant_id: str, user_id: Optional[str]) -> List[Dict[str, str]]:
    uid = str(user_id or "").strip()
    if not uid:
        return []
    try:
        profile = get_customer_profile(tenant_id=tenant_id, user_id=uid)
        raw_facts = list_memory_facts(
            tenant_id=tenant_id,
            user_id=uid,
            limit=max(20, int(MEMORY_LONGTERM_MAX_FACTS) * 4),
        )
    except Exception as e:
        logger.warning("Long-term memory load failed tenant=%s user=%s error=%s", tenant_id, uid, e)
        return []

    facts: List[Dict[str, Any]] = []
    min_conf = float(MEMORY_LONGTERM_MIN_CONFIDENCE)
    max_facts = max(1, int(MEMORY_LONGTERM_MAX_FACTS))
    for f in raw_facts:
        if not isinstance(f, dict):
            continue
        conf = f.get("confidence")
        if isinstance(conf, (int, float)) and float(conf) < min_conf:
            continue
        facts.append(f)
        if len(facts) >= max_facts:
            break

    if not profile and not facts:
        return []

    block = _format_long_term_memory_block(profile=profile, facts=facts)
    return [{"role": "system", "content": block}]


def _extract_pending_conflict(profile: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(profile, dict):
        return None
    pj = profile.get("profile_json")
    if not isinstance(pj, dict):
        return None
    c = pj.get("_memory_conflict")
    if not isinstance(c, dict):
        return None
    if str(c.get("status") or "").lower() != "pending":
        return None
    key = str(c.get("fact_key") or "").strip()
    options = [str(x).strip() for x in (c.get("options") or []) if str(x).strip()]
    if not key or len(options) < 2:
        return None
    out = dict(c)
    out["fact_key"] = key
    out["options"] = options[:2]
    return out


def _build_conflict_question(conflict: Dict[str, Any]) -> str:
    key = str(conflict.get("fact_key") or "").strip()
    options = [str(x).strip() for x in (conflict.get("options") or []) if str(x).strip()]
    if len(options) < 2:
        return "Anh/chị giúp em xác nhận lại thông tin gần nhất để em tư vấn chính xác hơn ạ."
    return (
        f"Để em tư vấn chính xác hơn, anh/chị xác nhận giúp em 1 lần: "
        f"{key} hiện tại là **{options[0]}** hay **{options[1]}** ạ?"
    )


def _mark_conflict_asked(*, tenant_id: str, user_id: str, conflict: Dict[str, Any]) -> None:
    c = dict(conflict or {})
    c["status"] = "pending"
    c["asked_once"] = True
    c["asked_at_ts"] = int(time.time())
    upsert_customer_profile(
        tenant_id=tenant_id,
        user_id=user_id,
        profile_patch={"_memory_conflict": c},
    )


def memory_rag_query(
    question: str,
    *,
    index: VectorStoreIndex,
    tenant_id: str,
    branch_id: Optional[str] = None,
    channel: str = "cli",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    stream: bool = False,
) -> Dict[str, object]:
    """
    Day 6-7 wrapper: persistent memory (Postgres) + optional roll-up summary (LLM).
    Returns the same dict shape as `agentic_query`.
    """
    if not MEMORY_ENABLED or not DATABASE_URL:
        return agentic_query(question, index=index, tenant_id=tenant_id, branch_id=branch_id, history=[], user_id=user_id)

    sid = (session_id or "").strip()
    if not sid:
        if not user_id:
            raise ValueError("memory_rag_query requires either session_id or user_id.")
        sid = build_session_id(tenant_id=tenant_id, channel=channel, user_id=str(user_id))

    # Ensure CLI convention tenant:session_id
    if ":" not in sid and tenant_id:
        sid = f"{tenant_id}:{sid}"
    if not sid.startswith(f"{tenant_id}:"):
        # Fail-closed to avoid accidental cross-tenant session load.
        sid = f"{tenant_id}:{sid}"

    state = get_or_create_session(session_id=sid, tenant_id=tenant_id)
    mem_ctx = build_history_from_session(state)
    profile_for_conflict = None
    if user_id:
        try:
            profile_for_conflict = get_customer_profile(tenant_id=tenant_id, user_id=str(user_id))
        except Exception:
            profile_for_conflict = None
    pending_conflict = _extract_pending_conflict(profile_for_conflict)
    if pending_conflict and not bool(pending_conflict.get("asked_once")):
        clarify_q = _build_conflict_question(pending_conflict)
        try:
            _mark_conflict_asked(tenant_id=tenant_id, user_id=str(user_id), conflict=pending_conflict)
        except Exception as e:
            logger.warning("Failed to mark conflict asked tenant=%s user=%s err=%s", tenant_id, user_id, e)
        state = update_session_after_turn(state=state, user_text=question, assistant_text=clarify_q, tool_metadata=None)
        _state2, roll_metrics = maybe_rollup_summary(state=state)
        return {
            "answer": clarify_q,
            "sources": [],
            "route": "memory_conflict_clarify",
            "memory": {
                "session_id": sid,
                "token_estimate": mem_ctx.token_estimate,
                "long_term_injected": False,
                "long_term_items": 0,
                "heuristic_extract_scheduled": False,
                "conflict_clarify_asked": True,
                "rolled_up": bool(roll_metrics.get("rolled_up")),
                "rollup_metrics": roll_metrics,
            },
        }

    long_term_history = _build_long_term_history(tenant_id=tenant_id, user_id=user_id)
    final_history = list(long_term_history)
    final_history.extend(mem_ctx.history)

    if stream:
        gen = agentic_query(
            question,
            index=index,
            tenant_id=tenant_id,
            branch_id=branch_id,
            history=final_history,
            user_id=user_id,
            stream=True,
        )
        def _memory_stream():
            final_result = {}
            for item in gen:
                if item["type"] == "result":
                    final_result = item["data"]
                else:
                    yield item
            
            # Post-process after stream
            answer = str(final_result.get("answer", "") or "")
            tool_md = final_result.get("tool_metadata") if isinstance(final_result.get("tool_metadata"), dict) else None
            state_after = update_session_after_turn(state=state, user_text=question, assistant_text=answer, tool_metadata=tool_md)

            # Roll up if needed
            state2, roll_metrics = maybe_rollup_summary(state=state_after)
            route = str(final_result.get("route") or "").strip() or None
            heuristic_scheduled = False
            try:
                heuristic_scheduled = submit_heuristic_extraction(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    session_id=sid,
                    route=route,
                    recent_messages=list(state2.recent_messages_buffer or []),
                    tool_metadata=tool_md,
                    trace_id=None,
                )
            except Exception as e:
                logger.warning("Failed to schedule heuristic extraction tenant=%s user=%s err=%s", tenant_id, user_id, e)

            final_result["memory"] = {
                "session_id": sid,
                "token_estimate": mem_ctx.token_estimate,
                "long_term_injected": bool(long_term_history),
                "long_term_items": len(long_term_history),
                "heuristic_extract_scheduled": bool(heuristic_scheduled),
                "rolled_up": bool(roll_metrics.get("rolled_up")),
                "rollup_metrics": roll_metrics,
            }
            yield {"type": "result", "data": final_result}
            
        return _memory_stream()

    result = agentic_query(
        question,
        index=index,
        tenant_id=tenant_id,
        branch_id=branch_id,
        history=final_history,
        user_id=user_id,
        stream=False,
    )

    answer = str(result.get("answer", "") or "")
    tool_md = result.get("tool_metadata") if isinstance(result.get("tool_metadata"), dict) else None
    state = update_session_after_turn(state=state, user_text=question, assistant_text=answer, tool_metadata=tool_md)

    # Roll up if needed
    state2, roll_metrics = maybe_rollup_summary(state=state)
    route = str(result.get("route") or "").strip() or None
    heuristic_scheduled = False
    try:
        heuristic_scheduled = submit_heuristic_extraction(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=sid,
            route=route,
            recent_messages=list(state2.recent_messages_buffer or []),
            tool_metadata=tool_md,
            trace_id=None,
        )
    except Exception as e:
        logger.warning("Failed to schedule heuristic extraction tenant=%s user=%s err=%s", tenant_id, user_id, e)

    result["memory"] = {
        "session_id": sid,
        "token_estimate": mem_ctx.token_estimate,
        "long_term_injected": bool(long_term_history),
        "long_term_items": len(long_term_history),
        "heuristic_extract_scheduled": bool(heuristic_scheduled),
        "rolled_up": bool(roll_metrics.get("rolled_up")),
        "rollup_metrics": roll_metrics,
    }
    return result
