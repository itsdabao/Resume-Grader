from __future__ import annotations

import hashlib
import json
import logging
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import (
    MEMORY_HEURISTIC_DEDUPE_TTL_SEC,
    MEMORY_HEURISTIC_ENABLED,
    MEMORY_HEURISTIC_MAX_MESSAGES,
    MEMORY_HEURISTIC_WORKERS,
    MEMORY_LLM_ASK_ON_CONFLICT,
    MEMORY_LLM_CONFIRM_THRESHOLD,
    MEMORY_LLM_CONFLICT_MARGIN,
    MEMORY_LLM_VERIFY_ENABLED,
    MEMORY_LLM_VERIFY_MAX_CANDIDATES,
)

from .extractor import heuristic_extract
from .store import (
    add_memory_event,
    add_memory_fact,
    get_customer_profile,
    list_memory_facts,
    soft_delete_memory_facts,
    upsert_customer_profile,
)
from .verifier import verify_candidates_with_llm


logger = logging.getLogger(__name__)

_WORKERS = max(1, int(MEMORY_HEURISTIC_WORKERS))
_EXECUTOR = ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="memory_hook2")
_SEEN_TURNS: Dict[str, float] = {}
_SEEN_LOCK = Lock()


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(obj)


def _normalize_fact_value(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _norm_text(value: str) -> str:
    t = unicodedata.normalize("NFD", str(value or ""))
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = t.replace("đ", "d").replace("Đ", "D")
    return _normalize_fact_value(t)


def _turn_fingerprint(
    *,
    tenant_id: str,
    user_id: str,
    session_id: Optional[str],
    route: Optional[str],
    recent_messages: List[Dict[str, Any]],
    tool_metadata: Dict[str, Any] | None,
) -> str:
    payload = {
        "tenant_id": str(tenant_id or ""),
        "user_id": str(user_id or ""),
        "session_id": str(session_id or ""),
        "route": str(route or ""),
        "recent_messages": [
            {
                "role": str(m.get("role") or ""),
                "content": str(m.get("content") or ""),
            }
            for m in (recent_messages or [])
            if isinstance(m, dict)
        ],
        "tool_metadata": tool_metadata if isinstance(tool_metadata, dict) else {},
    }
    raw = _safe_json_dumps(payload).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def _dedupe_turn_recently_seen(turn_hash: str) -> bool:
    now = time.time()
    ttl = max(60, int(MEMORY_HEURISTIC_DEDUPE_TTL_SEC))
    with _SEEN_LOCK:
        stale = [k for k, ts in _SEEN_TURNS.items() if (now - float(ts)) > ttl]
        for k in stale:
            _SEEN_TURNS.pop(k, None)
        if turn_hash in _SEEN_TURNS:
            return True
        _SEEN_TURNS[turn_hash] = now
        return False


def _fact_identity(f: Dict[str, Any]) -> Tuple[str, str]:
    key = str(f.get("fact_key") or "").strip().lower()
    val = str(f.get("fact_value_text") or "").strip()
    if not val and isinstance(f.get("fact_value_json"), dict):
        val = _safe_json_dumps(f.get("fact_value_json") or {})
    return key, _normalize_fact_value(val)


def _candidate_from_fact(f: Dict[str, Any]) -> Dict[str, Any]:
    val_text = str(f.get("fact_value_text") or "").strip()
    if not val_text and isinstance(f.get("fact_value_json"), dict):
        val_text = _safe_json_dumps(f.get("fact_value_json") or {})
    return {
        "candidate_id": str(f.get("id") or ""),
        "fact_key": str(f.get("fact_key") or "").strip(),
        "fact_value_text": val_text,
        "confidence": float(f.get("confidence") or 0.0),
    }


def _extract_profile_json(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    pj = profile.get("profile_json")
    return dict(pj) if isinstance(pj, dict) else {}


def _get_pending_conflict(profile_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    c = profile_json.get("_memory_conflict") if isinstance(profile_json, dict) else None
    if not isinstance(c, dict):
        return None
    if str(c.get("status") or "").lower() != "pending":
        return None
    key = str(c.get("fact_key") or "").strip()
    options = c.get("options") if isinstance(c.get("options"), list) else []
    vals = [str(x).strip() for x in options if str(x).strip()]
    if not key or not vals:
        return None
    out = dict(c)
    out["fact_key"] = key
    out["options"] = vals
    return out


def _set_pending_conflict(
    *,
    tenant_id: str,
    user_id: str,
    fact_key: str,
    options: List[str],
    existing_profile_json: Dict[str, Any],
) -> None:
    opts = [str(x).strip() for x in options if str(x).strip()]
    if len(opts) < 2:
        return
    pending_old = _get_pending_conflict(existing_profile_json)
    asked_once = False
    if pending_old:
        old_key = str(pending_old.get("fact_key") or "").strip()
        old_opts = sorted([_norm_text(x) for x in (pending_old.get("options") or []) if str(x).strip()])
        new_opts = sorted([_norm_text(x) for x in opts])
        if old_key == str(fact_key) and old_opts == new_opts:
            asked_once = bool(pending_old.get("asked_once"))

    upsert_customer_profile(
        tenant_id=tenant_id,
        user_id=user_id,
        profile_patch={
            "_memory_conflict": {
                "status": "pending",
                "fact_key": str(fact_key),
                "options": opts,
                "asked_once": bool(asked_once),
                "updated_at_ts": int(time.time()),
            }
        },
    )


def _clear_pending_conflict(*, tenant_id: str, user_id: str, fact_key: str, resolved_value: str) -> None:
    upsert_customer_profile(
        tenant_id=tenant_id,
        user_id=user_id,
        profile_patch={
            "_memory_conflict": {
                "status": "resolved",
                "fact_key": str(fact_key),
                "resolved_value": str(resolved_value),
                "asked_once": False,
                "resolved_at_ts": int(time.time()),
            }
        },
    )


def _latest_user_message_text(recent_messages: List[Dict[str, Any]]) -> str:
    for m in reversed(list(recent_messages or [])):
        if not isinstance(m, dict):
            continue
        if str(m.get("role") or "").lower() != "user":
            continue
        txt = str(m.get("content") or "").strip()
        if txt:
            return txt
    return ""


def _match_pending_conflict_reply(pending: Dict[str, Any], recent_messages: List[Dict[str, Any]]) -> Optional[str]:
    txt = _latest_user_message_text(recent_messages)
    if not txt:
        return None
    n_txt = _norm_text(txt)
    matches: List[str] = []
    for opt in pending.get("options") or []:
        o = str(opt).strip()
        if not o:
            continue
        n_opt = _norm_text(o)
        if n_opt and n_opt in n_txt:
            matches.append(o)
            continue
        # Numeric fallback for score-like options, e.g. "IELTS 6.5"
        m_num = None
        try:
            m_num = o.split()[-1]
        except Exception:
            m_num = None
        if m_num and str(m_num) in txt:
            matches.append(o)
    uniq = list(dict.fromkeys(matches))
    if len(uniq) == 1:
        return uniq[0]
    return None


def _group_values_by_key(facts: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for f in facts:
        if not isinstance(f, dict):
            continue
        key = str(f.get("fact_key") or "").strip()
        if not key:
            continue
        val = str(f.get("fact_value_text") or "").strip()
        if not val and isinstance(f.get("fact_value_json"), dict):
            val = _safe_json_dumps(f.get("fact_value_json") or {})
        if not val:
            continue
        conf = float(f.get("confidence") or 0.0)
        bucket = out.setdefault(key, {})
        prev = float(bucket.get(val) or 0.0)
        if conf > prev:
            bucket[val] = conf
    return out


def run_llm_verification_sync(
    *,
    tenant_id: str,
    user_id: str,
    session_id: Optional[str],
    route: Optional[str],
    recent_messages: List[Dict[str, Any]],
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not MEMORY_LLM_VERIFY_ENABLED:
        return {"ok": False, "reason": "disabled"}

    tnt = str(tenant_id or "").strip()
    uid = str(user_id or "").strip()
    if not tnt or not uid:
        return {"ok": False, "reason": "missing_scope"}

    profile = get_customer_profile(tenant_id=tnt, user_id=uid)
    if isinstance(profile, dict) and str(profile.get("consent_status") or "").lower() == "revoked":
        return {"ok": False, "reason": "consent_revoked"}

    profile_json = _extract_profile_json(profile)
    pending = _get_pending_conflict(profile_json)

    # Rule: if pending conflict exists and user answered clearly, resolve immediately.
    if pending:
        chosen = _match_pending_conflict_reply(pending, recent_messages)
        if chosen:
            fact_key = str(pending.get("fact_key") or "").strip()
            if fact_key:
                soft_delete_memory_facts(tenant_id=tnt, user_id=uid, fact_key=fact_key)
                add_memory_fact(
                    tenant_id=tnt,
                    user_id=uid,
                    fact_key=fact_key,
                    fact_value_text=str(chosen),
                    fact_value_json={"_meta": {"source": "conflict_user_clarification"}},
                    confidence=0.95,
                    session_id=(str(session_id) if session_id else None),
                    source_route=(str(route) if route else "memory_conflict_clarify"),
                    source_trace_id=(str(trace_id) if trace_id else None),
                )
                _clear_pending_conflict(tenant_id=tnt, user_id=uid, fact_key=fact_key, resolved_value=str(chosen))
                add_memory_event(
                    tenant_id=tnt,
                    user_id=uid,
                    session_id=(str(session_id) if session_id else None),
                    source_route=(str(route) if route else None),
                    source_trace_id=(str(trace_id) if trace_id else None),
                    event_type="memory_conflict_resolved",
                    content_text=f"{fact_key}={chosen}",
                    payload={"resolved_by": "user_reply", "options": pending.get("options") or []},
                )
                return {
                    "ok": True,
                    "reason": "conflict_resolved_by_user",
                    "resolved_key": fact_key,
                    "resolved_value": str(chosen),
                }

    facts = list_memory_facts(tenant_id=tnt, user_id=uid, limit=300)
    if not facts:
        return {"ok": True, "reason": "no_facts"}

    confirm_threshold = float(MEMORY_LLM_CONFIRM_THRESHOLD)
    candidates_raw = [
        f
        for f in facts
        if isinstance(f, dict)
        and isinstance(f.get("confidence"), (int, float))
        and float(f.get("confidence") or 0.0) < confirm_threshold
    ]
    candidates_raw = candidates_raw[: max(1, int(MEMORY_LLM_VERIFY_MAX_CANDIDATES))]
    candidates = [_candidate_from_fact(f) for f in candidates_raw]
    decisions = verify_candidates_with_llm(recent_messages=recent_messages, candidates=candidates)

    promoted = 0
    rejected = 0
    for c in candidates_raw:
        cid = str(c.get("id") or "").strip()
        if not cid:
            continue
        d = decisions.get(cid)
        if not isinstance(d, dict):
            continue
        decision = str(d.get("decision") or "").lower().strip()
        conf = float(d.get("confidence") or 0.0)
        fact_key = str(c.get("fact_key") or "").strip()
        fact_text = str(c.get("fact_value_text") or "").strip()
        if not fact_key or not fact_text:
            continue

        if decision == "confirm" and conf >= confirm_threshold:
            add_memory_fact(
                tenant_id=tnt,
                user_id=uid,
                fact_key=fact_key,
                fact_value_text=fact_text,
                fact_value_json={"_meta": {"source": "llm_verify", "reason": str(d.get("reason") or "")}},
                confidence=conf,
                session_id=(str(session_id) if session_id else None),
                source_route=(str(route) if route else "memory_llm_verify"),
                source_trace_id=(str(trace_id) if trace_id else None),
            )
            promoted += 1
        elif decision == "reject":
            soft_delete_memory_facts(
                tenant_id=tnt,
                user_id=uid,
                fact_key=fact_key,
                fact_value_text=fact_text,
            )
            rejected += 1

    facts2 = list_memory_facts(tenant_id=tnt, user_id=uid, limit=300)
    grouped = _group_values_by_key(facts2)

    auto_resolved = 0
    pending_conflicts = 0
    margin = float(MEMORY_LLM_CONFLICT_MARGIN)
    for fact_key, value_conf_map in grouped.items():
        if len(value_conf_map) <= 1:
            continue
        ranked = sorted(value_conf_map.items(), key=lambda x: float(x[1]), reverse=True)
        if len(ranked) < 2:
            continue
        top_val, top_conf = ranked[0]
        second_val, second_conf = ranked[1]
        if float(top_conf) - float(second_conf) >= margin:
            # Rule: keep highest-confidence value
            for loser_val, _ in ranked[1:]:
                soft_delete_memory_facts(
                    tenant_id=tnt,
                    user_id=uid,
                    fact_key=fact_key,
                    fact_value_text=str(loser_val),
                )
            _clear_pending_conflict(tenant_id=tnt, user_id=uid, fact_key=fact_key, resolved_value=str(top_val))
            auto_resolved += 1
        else:
            # Rule: if cannot decide, ask customer once.
            if MEMORY_LLM_ASK_ON_CONFLICT:
                options = [str(v) for v, _ in ranked[:2]]
                _set_pending_conflict(
                    tenant_id=tnt,
                    user_id=uid,
                    fact_key=fact_key,
                    options=options,
                    existing_profile_json=profile_json,
                )
                pending_conflicts += 1

    try:
        add_memory_event(
            tenant_id=tnt,
            user_id=uid,
            session_id=(str(session_id) if session_id else None),
            source_route=(str(route) if route else None),
            source_trace_id=(str(trace_id) if trace_id else None),
            event_type="llm_verify",
            content_text=f"promoted={promoted}, rejected={rejected}",
            payload={
                "promoted": int(promoted),
                "rejected": int(rejected),
                "auto_resolved": int(auto_resolved),
                "pending_conflicts": int(pending_conflicts),
            },
        )
    except Exception as e:
        logger.warning("Failed to insert llm_verify event: %s", e)

    return {
        "ok": True,
        "reason": "verified",
        "promoted": int(promoted),
        "rejected": int(rejected),
        "auto_resolved": int(auto_resolved),
        "pending_conflicts": int(pending_conflicts),
    }


def run_heuristic_pipeline_sync(
    *,
    tenant_id: str,
    user_id: str,
    session_id: Optional[str],
    route: Optional[str],
    recent_messages: List[Dict[str, Any]],
    tool_metadata: Dict[str, Any] | None = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deterministic post-turn memory pipeline:
    Hook 2A (heuristic) + Hook 2B (llm verify/conflict resolver).
    """
    if not MEMORY_HEURISTIC_ENABLED and not MEMORY_LLM_VERIFY_ENABLED:
        return {"ok": False, "reason": "disabled"}

    tnt = str(tenant_id or "").strip()
    uid = str(user_id or "").strip()
    if not tnt or not uid:
        return {"ok": False, "reason": "missing_scope"}

    profile = get_customer_profile(tenant_id=tnt, user_id=uid)
    if isinstance(profile, dict) and str(profile.get("consent_status") or "").lower() == "revoked":
        return {"ok": False, "reason": "consent_revoked"}

    clipped_msgs = list(recent_messages or [])[-max(2, int(MEMORY_HEURISTIC_MAX_MESSAGES)) :]
    turn_hash = _turn_fingerprint(
        tenant_id=tnt,
        user_id=uid,
        session_id=session_id,
        route=route,
        recent_messages=clipped_msgs,
        tool_metadata=tool_metadata,
    )
    if _dedupe_turn_recently_seen(turn_hash):
        return {"ok": True, "reason": "duplicate_turn", "turn_hash": turn_hash, "saved_facts": 0}

    profile_patch: Dict[str, Any] = {}
    facts: List[Dict[str, Any]] = []
    if MEMORY_HEURISTIC_ENABLED:
        profile_patch, facts = heuristic_extract(
            tenant_id=tnt,
            user_id=uid,
            recent_messages=clipped_msgs,
            tool_metadata=tool_metadata,
            route=route,
        )

    if profile_patch:
        upsert_customer_profile(tenant_id=tnt, user_id=uid, profile_patch=profile_patch)

    existing = list_memory_facts(tenant_id=tnt, user_id=uid, limit=200)
    seen = {_fact_identity(f) for f in existing if isinstance(f, dict)}
    now_ts = int(time.time())
    saved_facts = 0
    skipped_dup = 0
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        fact_key = str(fact.get("fact_key") or "").strip()
        fact_text = str(fact.get("fact_value_text") or "").strip()
        fact_json = fact.get("fact_value_json") if isinstance(fact.get("fact_value_json"), dict) else {}
        confidence = fact.get("confidence")
        source = str(fact.get("source") or "heuristic")
        ttl_days = int(fact.get("ttl_days") or 0)
        if not fact_key:
            continue
        if not fact_text and fact_json:
            fact_text = _safe_json_dumps(fact_json)
        ident = (fact_key.lower(), _normalize_fact_value(fact_text))
        if ident in seen:
            skipped_dup += 1
            continue
        expires_at_ts = int(now_ts + ttl_days * 86400) if ttl_days > 0 else None
        value_json = dict(fact_json or {})
        value_json["_meta"] = {
            "extractor": "heuristic_v1",
            "source": source,
            "turn_hash": turn_hash,
        }
        add_memory_fact(
            tenant_id=tnt,
            user_id=uid,
            fact_key=fact_key,
            fact_value_text=fact_text,
            fact_value_json=value_json,
            confidence=(float(confidence) if isinstance(confidence, (int, float)) else None),
            session_id=(str(session_id) if session_id else None),
            source_route=(str(route) if route else None),
            source_trace_id=(str(trace_id) if trace_id else None),
            expires_at_ts=expires_at_ts,
        )
        seen.add(ident)
        saved_facts += 1

    try:
        add_memory_event(
            tenant_id=tnt,
            user_id=uid,
            session_id=(str(session_id) if session_id else None),
            source_route=(str(route) if route else None),
            source_trace_id=(str(trace_id) if trace_id else None),
            event_type="heuristic_extract",
            content_text=f"saved_facts={saved_facts}",
            payload={
                "profile_keys": sorted(list(profile_patch.keys())),
                "facts_extracted": len(facts),
                "saved_facts": int(saved_facts),
                "skipped_dup": int(skipped_dup),
                "turn_hash": turn_hash,
            },
        )
    except Exception as e:
        logger.warning("Failed to insert memory event: %s", e)

    llm_out: Dict[str, Any] | None = None
    if MEMORY_LLM_VERIFY_ENABLED:
        llm_out = run_llm_verification_sync(
            tenant_id=tnt,
            user_id=uid,
            session_id=session_id,
            route=route,
            recent_messages=clipped_msgs,
            trace_id=trace_id,
        )

    return {
        "ok": True,
        "reason": "saved",
        "turn_hash": turn_hash,
        "saved_profile": bool(profile_patch),
        "saved_facts": int(saved_facts),
        "skipped_dup": int(skipped_dup),
        "llm_verify": llm_out,
    }


def submit_heuristic_extraction(
    *,
    tenant_id: str,
    user_id: Optional[str],
    session_id: Optional[str],
    route: Optional[str],
    recent_messages: List[Dict[str, Any]],
    tool_metadata: Dict[str, Any] | None = None,
    trace_id: Optional[str] = None,
) -> bool:
    """
    Fire-and-forget post-turn memory scheduler.
    """
    if not MEMORY_HEURISTIC_ENABLED and not MEMORY_LLM_VERIFY_ENABLED:
        return False
    uid = str(user_id or "").strip()
    if not uid:
        return False
    clipped_msgs = list(recent_messages or [])[-max(2, int(MEMORY_HEURISTIC_MAX_MESSAGES)) :]
    fut = _EXECUTOR.submit(
        run_heuristic_pipeline_sync,
        tenant_id=str(tenant_id or "").strip(),
        user_id=uid,
        session_id=(str(session_id) if session_id else None),
        route=(str(route) if route else None),
        recent_messages=clipped_msgs,
        tool_metadata=(tool_metadata if isinstance(tool_metadata, dict) else None),
        trace_id=(str(trace_id) if trace_id else None),
    )

    def _done_cb(f):
        try:
            _ = f.result()
        except Exception as e:
            logger.warning("Memory background task failed: %s", e)

    fut.add_done_callback(_done_cb)
    return True
