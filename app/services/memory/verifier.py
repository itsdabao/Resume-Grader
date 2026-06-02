from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from llama_index.core import Settings

from app.core.config import MEMORY_LLM_CONFIRM_THRESHOLD, MEMORY_LLM_VERIFY_MAX_MESSAGES


logger = logging.getLogger(__name__)


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        raw = m.group(1).strip()
    if not raw.startswith("{"):
        m2 = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
        if m2:
            raw = m2.group(1).strip()
    return json.loads(raw)


def _call_llm(prompt: str) -> str:
    # Avoid lazy-loading default OpenAI model when runtime has not bootstrapped an LLM yet.
    llm = getattr(Settings, "_llm", None)
    if llm is None:
        raise RuntimeError("LLM not initialized")
    resp = llm.complete(prompt)
    txt = getattr(resp, "text", None)
    return str(txt if txt is not None else resp)


def _messages_to_text(messages: List[Dict[str, Any]], *, max_messages: int) -> str:
    lines: List[str] = []
    for m in (messages or [])[-max(2, int(max_messages)) :]:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip().lower()
        content = str(m.get("content") or "").strip()
        if not content or role not in ("user", "assistant"):
            continue
        tag = "User" if role == "user" else "Assistant"
        lines.append(f"{tag}: {content}")
    return "\n".join(lines)


def build_verify_prompt(
    *,
    recent_messages: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> str:
    transcript = _messages_to_text(recent_messages, max_messages=MEMORY_LLM_VERIFY_MAX_MESSAGES)
    cand = json.dumps(candidates, ensure_ascii=False)
    return (
        "ROLE: Bạn là bộ kiểm duyệt trí nhớ hội thoại.\n"
        "Mục tiêu: xác định fact nào đáng tin để lưu long-term memory.\n"
        "Nguyên tắc:\n"
        "- Chỉ dựa vào hội thoại được cung cấp.\n"
        "- Nếu chắc chắn user khẳng định rõ -> decision=confirm.\n"
        "- Nếu user phủ định/không liên quan -> decision=reject.\n"
        "- Nếu chưa đủ chắc chắn -> decision=uncertain.\n"
        "- confidence trong [0,1].\n"
        f"- Chỉ confirm khi confidence >= {float(MEMORY_LLM_CONFIRM_THRESHOLD):.2f}.\n"
        "Trả về DUY NHẤT một JSON object theo schema:\n"
        "{\n"
        '  "decisions": [\n'
        "    {\n"
        '      "candidate_id": "string",\n'
        '      "decision": "confirm|reject|uncertain",\n'
        '      "confidence": 0.0,\n'
        '      "normalized_value": "string|null",\n'
        '      "reason": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "INPUT:\n"
        f"- Transcript:\n{transcript}\n"
        f"- Candidates JSON:\n{cand}\n"
    )


def verify_candidates_with_llm(
    *,
    recent_messages: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Returns map: candidate_id -> {decision, confidence, normalized_value, reason}
    """
    if not candidates:
        return {}
    try:
        prompt = build_verify_prompt(recent_messages=recent_messages, candidates=candidates)
        out_text = _call_llm(prompt)
        obj = _extract_json_object(out_text)
    except Exception as e:
        if "LLM not initialized" in str(e):
            logger.debug("LLM verifier skipped: %s", e)
        else:
            logger.warning("LLM verifier failed: %s", e)
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    rows = obj.get("decisions") if isinstance(obj.get("decisions"), list) else []
    for r in rows:
        if not isinstance(r, dict):
            continue
        cid = str(r.get("candidate_id") or "").strip()
        if not cid:
            continue
        decision = str(r.get("decision") or "").strip().lower()
        if decision not in ("confirm", "reject", "uncertain"):
            decision = "uncertain"
        conf_raw = r.get("confidence")
        try:
            conf = float(conf_raw)
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        norm_val = r.get("normalized_value")
        out[cid] = {
            "decision": decision,
            "confidence": conf,
            "normalized_value": (str(norm_val).strip() if isinstance(norm_val, str) and str(norm_val).strip() else None),
            "reason": str(r.get("reason") or "").strip(),
        }
    return out
