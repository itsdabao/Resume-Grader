"""
Centralized pre-router security filter.

Runs BEFORE route_query() in the agentic pipeline to block:
- Prompt injection / system prompt leakage attempts
- Social engineering (authority claims, bypass requests)
- Data exfiltration (requests for internal files, DB, API keys)

Design: pure regex/keyword – no LLM call, no embedding, < 1ms.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SecurityDecision:
    blocked: bool
    category: str          # "prompt_leak" | "social_engineering" | "data_exfil" | "safe"
    reason: str
    safe_response: Optional[str] = None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    t = unicodedata.normalize("NFD", text or "")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = t.replace("đ", "d").replace("Đ", "D").lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


# ---------------------------------------------------------------------------
# Pattern banks (Vietnamese + English)
# ---------------------------------------------------------------------------

_PROMPT_LEAK_PATTERNS: List[str] = [
    # Vietnamese
    r"\bbo qua\b.*\bhuong dan\b",
    r"\bbo qua\b.*\bluat\b",
    r"\bbo qua\b.*\bquy tac\b",
    r"\bin ra\b.*\bsystem prompt\b",
    r"\bin ra\b.*\bhuong dan he thong\b",
    r"\bhien thi\b.*\bprompt\b",
    r"\bcho xem\b.*\bprompt\b",
    r"\bnoi dung\b.*\bsystem\b",
    r"\bhuong dan noi bo\b",
    r"\bchu dan noi bo\b",
    r"\bliet ke\b.*\bquy tac\b",
    # English
    r"\bignore\b.*\b(?:previous|above|all)\b.*\b(?:instruction|rule|prompt)\b",
    r"\bprint\b.*\bsystem prompt\b",
    r"\bshow\b.*\bsystem prompt\b",
    r"\breveal\b.*\b(?:instruction|prompt)\b",
    r"\bdisregard\b.*\b(?:instruction|rule|guideline|prompt)\b",
    r"\bdisregard\b.*\b(?:above|previous|all)\b",
    r"\bforget\b.*\b(?:instruction|rule|everything)\b",
    r"\byou are now\b",
    r"\bact as\b.*\b(?:developer|admin|root)\b",
    r"\bjailbreak\b",
    r"\bdan mode\b",
]

_SOCIAL_ENGINEERING_PATTERNS: List[str] = [
    r"\bgiam doc\b",
    r"\bquản lý\b",
    r"\bquan ly\b",
    r"\badmin\b.*\bbypass\b",
    r"\bbypass\b",
    r"\bvip\b.*\b(?:admin|code|password|mat khau)\b",
    r"\bmat khau\b",
    r"\bpassword\b",
    r"\bxac nhan\b.*\bgiam gia\b.*\b(?:1 vnd|0 dong|mien phi)\b",
    r"\bgiam gia\b.*\bcon\b.*\b(?:1 vnd|0 dong|1vnd|0dong)\b",
    r"\btoken\b.*\b(?:admin|bypass|master)\b",
    r"\bma\b.*\bbypass\b",
    r"\bem hay\b.*\bxac nhan\b.*\bgiam\b",
    r"\banh la\b.*\b(?:giam doc|sep|admin|quan ly)\b",
    r"\btoi la\b.*\b(?:giam doc|sep|admin|quan ly)\b",
]

_DATA_EXFIL_PATTERNS: List[str] = [
    r"\bnodes\.jsonl\b",
    r"\bdatabase\b.*\b(?:schema|dump|export)\b",
    r"\bapi[_ ]?key\b",
    r"\benv\b.*\bfile\b",
    r"\bfile\b.*\benv\b",
    r"\bexport\b.*\b(?:data|log|trace|user)\b",
    r"\bshow\b.*\b(?:data|log|trace|secret)\b",
    r"\bdump\b.*\b(?:data|memory|user|table)\b",
    r"\bsql\b.*\b(?:inject|query|select|drop)\b",
    r"\bselect\b.*\bfrom\b",
    r"\bdrop table\b",
]


# ---------------------------------------------------------------------------
# Safe responses
# ---------------------------------------------------------------------------

_SAFE_RESPONSES = {
    "prompt_leak": (
        "Dạ em không thể chia sẻ nội dung hướng dẫn hệ thống ạ. "
        "Anh/chị có câu hỏi gì về khoá học thì em sẵn sàng hỗ trợ nhé!"
    ),
    "social_engineering": (
        "Dạ em không có quyền thay đổi chính sách giá hay xác nhận yêu cầu đặc biệt ạ. "
        "Mọi ưu đãi đều theo chính sách chính thức của trung tâm. "
        "Anh/chị cần tư vấn thêm thì em kết nối với bộ phận hỗ trợ nhé!"
    ),
    "data_exfil": (
        "Dạ em không thể cung cấp thông tin kỹ thuật nội bộ ạ. "
        "Anh/chị cho em biết khoá học cần tìm hiểu để em hỗ trợ nhé!"
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check(question: str) -> SecurityDecision:
    """
    Evaluate a user query for security threats.
    Returns SecurityDecision with blocked=True if the query should be rejected.
    """
    if not question or not question.strip():
        return SecurityDecision(blocked=False, category="safe", reason="empty_query")

    qn = _norm(question)
    q_lower = (question or "").lower()

    # --- Raw-text checks for patterns destroyed by normalization ---
    _RAW_BLOCK_PATTERNS = [
        (".env", "data_exfil"),
        ("select *", "data_exfil"),
        ("select * from", "data_exfil"),
    ]
    for raw_pat, raw_cat in _RAW_BLOCK_PATTERNS:
        if raw_pat in q_lower:
            logger.warning("security_filter BLOCKED %s: raw_pattern=%s query=%s", raw_cat, raw_pat, question[:120])
            return SecurityDecision(
                blocked=True,
                category=raw_cat,
                reason=f"raw_matched:{raw_pat}",
                safe_response=_SAFE_RESPONSES[raw_cat],
            )

    # 1) Prompt leak / jailbreak
    for pat in _PROMPT_LEAK_PATTERNS:
        if re.search(pat, qn):
            logger.warning("security_filter BLOCKED prompt_leak: pattern=%s query=%s", pat, question[:120])
            return SecurityDecision(
                blocked=True,
                category="prompt_leak",
                reason=f"matched:{pat}",
                safe_response=_SAFE_RESPONSES["prompt_leak"],
            )

    # 2) Social engineering
    for pat in _SOCIAL_ENGINEERING_PATTERNS:
        if re.search(pat, qn):
            logger.warning("security_filter BLOCKED social_engineering: pattern=%s query=%s", pat, question[:120])
            return SecurityDecision(
                blocked=True,
                category="social_engineering",
                reason=f"matched:{pat}",
                safe_response=_SAFE_RESPONSES["social_engineering"],
            )

    # 3) Data exfiltration
    for pat in _DATA_EXFIL_PATTERNS:
        if re.search(pat, qn):
            logger.warning("security_filter BLOCKED data_exfil: pattern=%s query=%s", pat, question[:120])
            return SecurityDecision(
                blocked=True,
                category="data_exfil",
                reason=f"matched:{pat}",
                safe_response=_SAFE_RESPONSES["data_exfil"],
            )

    return SecurityDecision(blocked=False, category="safe", reason="passed")
