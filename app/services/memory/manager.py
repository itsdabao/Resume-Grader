from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core import Settings

from app.core.config import (
    MEMORY_BUDGET_TOKENS,
    MEMORY_LAST_TURNS,
    MEMORY_SUMMARY_ENABLED,
    MEMORY_SUMMARY_MAX_OUTPUT_TOKENS,
)
from app.services.agentic.preprocess import extract_phone

from .store import SessionState, append_messages, merge_entity_memory, save_session


def estimate_tokens_char4(s: str) -> int:
    # Cheap heuristic: 1 token ~ 4 chars in Latin-ish text.
    return int(max(0, len(s or "")) / 4)


def _messages_to_text(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        tag = "User" if role == "user" else "Assistant"
        lines.append(f"{tag}: {content}")
    return "\n".join(lines)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON extraction:
    - supports code blocks ```json ... ```
    - extracts the first {...} object
    """
    raw = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
    # Fallback: first {...}
    if not raw.startswith("{"):
        m2 = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
        if m2:
            raw = m2.group(1).strip()
    return json.loads(raw)


def _call_llm(prompt: str) -> str:
    llm = Settings.llm
    if llm is None:
        raise RuntimeError("LLM is not initialized (Settings.llm is None).")
    resp = llm.complete(prompt)
    # llama-index completion response usually exposes `.text`
    txt = getattr(resp, "text", None)
    return str(txt if txt is not None else resp)


def build_summary_prompt(*, prev_summary: str, messages_text: str, entity_memory: Dict[str, Any]) -> str:
    """
    Ask LLM to produce rolling summary + entity_memory patch.
    Output must be a single JSON object.
    """
    entity_json = json.dumps(entity_memory or {}, ensure_ascii=False)
    return (
        "ROLE: Bạn là hệ thống tóm tắt hội thoại cho chatbot tư vấn trung tâm Anh ngữ.\n"
        "Mục tiêu: tạo 'Rolling Summary' ngắn gọn nhưng đầy đủ để chatbot nhớ đúng ngữ cảnh.\n"
        "\n"
        "YÊU CẦU:\n"
        "- Chỉ dùng thông tin trong hội thoại.\n"
        "- Ưu tiên giữ số liệu quan trọng (học phí, giảm giá, tổng thanh toán, thời lượng, lịch khai giảng, SĐT).\n"
        "- Tránh chi tiết thừa; tập trung vào nhu cầu, khóa quan tâm, ràng buộc, các con số.\n"
        "- Trả về DUY NHẤT 1 JSON object hợp lệ (không kèm giải thích).\n"
        "\n"
        "ĐẦU VÀO:\n"
        f"- Previous rolling_summary:\n{prev_summary.strip()}\n"
        f"- Current entity_memory (JSON):\n{entity_json}\n"
        f"- Messages to roll up:\n{messages_text.strip()}\n"
        "\n"
        "OUTPUT JSON SCHEMA:\n"
        "{\n"
        '  "rolling_summary": "string",\n'
        '  "entity_memory_patch": {\n'
        '    "phone": "string|null",\n'
        '    "intent": "string|null",\n'
        '    "course_interest": "string|null",\n'
        '    "computed_total_payable_vnd": "number|null",\n'
        '    "discount_scope": "tuition|total|null"\n'
        "  }\n"
        "}\n"
    )


def _heuristic_entity_patch(user_text: str, assistant_text: str, tool_metadata: Dict[str, Any] | None) -> Dict[str, Any]:
    patch: Dict[str, Any] = {}
    phone = extract_phone(user_text or "")
    if phone:
        patch["phone"] = phone
    md = tool_metadata or {}
    # Capture business-critical numbers in structured form when present.
    if isinstance(md.get("computed_final_vnd"), (int, float)):
        patch["computed_total_payable_vnd"] = int(md["computed_final_vnd"])
    if isinstance(md.get("discount_scope"), str):
        patch["discount_scope"] = md["discount_scope"]
    return patch


@dataclass
class MemoryContext:
    history: List[Dict[str, str]]
    token_estimate: int


def build_history_from_session(state: SessionState) -> MemoryContext:
    """
    Build history for LLM prompt:
    - Always include rolling_summary (if any)
    - Include last N turns from buffer
    """
    max_msgs = max(0, int(MEMORY_LAST_TURNS) * 2)
    buf = list(state.recent_messages_buffer or [])
    if max_msgs and len(buf) > max_msgs:
        buf = buf[-max_msgs:]

    history: List[Dict[str, str]] = []
    if (state.rolling_summary or "").strip():
        history.append({"role": "assistant", "content": f"[Rolling Summary]\n{state.rolling_summary.strip()}"})
    for m in buf:
        role = (m.get("role") or "user").lower()
        content = str(m.get("content") or "")
        if role in ("user", "assistant") and content:
            history.append({"role": role, "content": content})

    token_est = estimate_tokens_char4(state.rolling_summary or "") + sum(estimate_tokens_char4(m.get("content", "")) for m in buf)
    return MemoryContext(history=history, token_estimate=token_est)


def maybe_rollup_summary(
    *,
    state: SessionState,
    budget_tokens: int = MEMORY_BUDGET_TOKENS,
    keep_turns: int = MEMORY_LAST_TURNS,
) -> Tuple[SessionState, Dict[str, Any]]:
    """
    If (rolling_summary + buffer) exceeds budget, roll up the older buffer into rolling_summary.
    Returns (updated_state, metrics).
    """
    metrics: Dict[str, Any] = {"rolled_up": False, "budget_tokens": int(budget_tokens)}
    if not MEMORY_SUMMARY_ENABLED:
        return state, metrics

    keep_msgs = max(0, int(keep_turns) * 2)
    buf = list(state.recent_messages_buffer or [])
    summary_tokens = estimate_tokens_char4(state.rolling_summary or "")
    buf_tokens = sum(estimate_tokens_char4(str(m.get("content") or "")) for m in buf)
    total_tokens = int(summary_tokens + buf_tokens)
    metrics["token_estimate_before"] = total_tokens
    metrics["buffer_messages_before"] = len(buf)

    if total_tokens <= int(budget_tokens):
        return state, metrics

    if keep_msgs <= 0:
        keep_msgs = 0

    old_msgs = buf[:-keep_msgs] if keep_msgs and len(buf) > keep_msgs else buf
    recent = buf[-keep_msgs:] if keep_msgs and len(buf) > keep_msgs else []

    # If nothing to roll up, just trim.
    if not old_msgs:
        state.recent_messages_buffer = recent
        metrics["rolled_up"] = False
        metrics["buffer_messages_after"] = len(recent)
        return state, metrics

    prompt = build_summary_prompt(
        prev_summary=state.rolling_summary or "",
        messages_text=_messages_to_text(old_msgs),
        entity_memory=state.entity_memory or {},
    )

    t0 = time.perf_counter()
    try:
        out_text = _call_llm(prompt)
        obj = _extract_json_object(out_text)
        new_summary = str(obj.get("rolling_summary") or "").strip()
        patch = obj.get("entity_memory_patch") if isinstance(obj.get("entity_memory_patch"), dict) else {}
        state.rolling_summary = new_summary
        state.entity_memory = merge_entity_memory(state.entity_memory or {}, patch or {})
        state.recent_messages_buffer = recent
        metrics["rolled_up"] = True
        metrics["llm_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    except Exception as e:
        # Fail-safe: don't crash the chat; just trim the buffer.
        state.recent_messages_buffer = recent
        metrics["rolled_up"] = False
        metrics["error"] = str(e)

    # Persist
    save_session(state=state)

    summary_tokens2 = estimate_tokens_char4(state.rolling_summary or "")
    buf_tokens2 = sum(estimate_tokens_char4(str(m.get("content") or "")) for m in state.recent_messages_buffer or [])
    metrics["token_estimate_after"] = int(summary_tokens2 + buf_tokens2)
    metrics["buffer_messages_after"] = len(state.recent_messages_buffer or [])
    return state, metrics


def update_session_after_turn(
    *,
    state: SessionState,
    user_text: str,
    assistant_text: str,
    tool_metadata: Optional[Dict[str, Any]] = None,
) -> SessionState:
    patch = _heuristic_entity_patch(user_text, assistant_text, tool_metadata)
    state.entity_memory = merge_entity_memory(state.entity_memory or {}, patch)
    state = append_messages(
        state=state,
        messages=[
            {"role": "user", "content": user_text, "ts": int(time.time())},
            {"role": "assistant", "content": assistant_text, "ts": int(time.time())},
        ],
        max_messages=max(0, int(MEMORY_LAST_TURNS) * 2 * 3),  # keep a bit more before rollup
    )
    save_session(state=state)
    return state

