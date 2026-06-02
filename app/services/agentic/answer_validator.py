"""
Post-Answer Validator — checks agent responses before returning to user.

Validates:
1. Tenant isolation: no cross-tenant data leakage
2. Numeric consistency: tool-computed numbers must match the answer text
3. Policy compliance: answer must not promise outside policy registry rules
4. Internal data leak: no system prompt fragments, JSON debug, file paths

Design: pure regex/heuristic — no LLM call, < 1ms.
If validation fails, returns a sanitized response or handoff.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    violations: List[str] = field(default_factory=list)
    sanitized_answer: Optional[str] = None
    action: str = "pass"  # "pass" | "sanitize" | "handoff"


# ---------------------------------------------------------------------------
# Internal data patterns that MUST NOT appear in answers
# ---------------------------------------------------------------------------

_INTERNAL_LEAK_PATTERNS = [
    r"system\s*prompt",
    r"```(?:json|python|yaml)",       # code blocks with language tags
    r"\bnode[s_]\.jsonl\b",
    r"\b(?:app|services|agentic|guardrails)[/\\]",  # file paths
    r"\b__[a-z_]+__\b",               # dunder attributes
    r"\bDEBUG\b.*\btrace\b",
    r"\btrace_id\b",
    r"\btenant_id\b.*=.*\b[a-z0-9_]{3,}\b",  # exposing tenant IDs
    r"\bapi[_\s]*key\b",
    r"\b(?:SELECT|INSERT|UPDATE|DELETE)\b.*\bFROM\b",
    r"\bSettings\.",                   # llama_index internal
    r"\bVectorStoreIndex\b",
    r"\bQdrantClient\b",
    r"\bembedding\b.*\bmodel\b",
    r"\"fact_key\"",                   # memory store internals
    r"\"fact_value_text\"",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_numbers_from_text(text: str) -> List[int]:
    """Extract all VND-scale numbers from text (ignore small numbers like percentages)."""
    numbers = []
    for m in re.finditer(r"([\d.,]+)\s*(?:VND|vnđ|đồng|dong|vnd|₫)", text, flags=re.IGNORECASE):
        raw = m.group(1).replace(".", "").replace(",", "")
        try:
            val = int(raw)
            if val >= 10_000:  # Only meaningful VND amounts
                numbers.append(val)
        except ValueError:
            pass
    return numbers


def _format_vnd(amount: int) -> str:
    """Format VND amount with dot separators."""
    return f"{amount:,}".replace(",", ".")


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def _check_internal_leak(answer: str) -> List[str]:
    """Check for internal data leakage in the answer."""
    violations = []
    for pat in _INTERNAL_LEAK_PATTERNS:
        if re.search(pat, answer, flags=re.IGNORECASE):
            violations.append(f"internal_leak:pattern={pat}")
    return violations


def _check_numeric_consistency(
    answer: str,
    tool_metadata: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Check that tool-computed numbers appear in the answer."""
    violations = []
    if not tool_metadata or not isinstance(tool_metadata, dict):
        return violations

    # Check computed_final_vnd from tuition calculator
    computed = tool_metadata.get("computed_final_vnd")
    if computed is not None and isinstance(computed, (int, float)):
        computed_int = int(computed)
        if computed_int > 0:
            # The number should appear in the answer (formatted or raw)
            answer_numbers = _extract_numbers_from_text(answer)
            raw_str = str(computed_int)
            formatted_str = _format_vnd(computed_int)

            found = (
                raw_str in answer
                or formatted_str in answer
                or computed_int in answer_numbers
            )
            if not found:
                violations.append(
                    f"numeric_mismatch:computed={computed_int} not found in answer"
                )

    return violations


def _check_tenant_isolation(
    answer: str,
    tenant_id: Optional[str] = None,
    all_tenant_ids: Optional[List[str]] = None,
) -> List[str]:
    """Check that the answer doesn't mention other tenants' data."""
    violations = []
    if not tenant_id or not all_tenant_ids:
        return violations

    current = str(tenant_id).strip().lower()
    for tid in all_tenant_ids:
        other = str(tid).strip().lower()
        if other and other != current and other in answer.lower():
            violations.append(f"cross_tenant_leak:mentioned={other}")

    return violations


def _check_forbidden_promises(answer: str) -> List[str]:
    """Check for promises that should not be made without verification."""
    violations = []
    answer_lower = answer.lower()

    forbidden_patterns = [
        (r"cam kết.*(?:100%|đậu|đạt|pass)", "promise:guarantee_result"),
        (r"miễn phí.*(?:toàn bộ|hoàn toàn|100%)", "promise:free_everything"),
        (r"hoàn.*100%.*học phí", "promise:full_refund"),
    ]

    for pat, violation_type in forbidden_patterns:
        if re.search(pat, answer_lower):
            violations.append(violation_type)

    return violations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_HANDOFF_RESPONSE = (
    "Dạ em cần xác nhận lại thông tin chính xác với bộ phận hỗ trợ. "
    "Anh/chị vui lòng để em chuyển cho tư vấn viên hỗ trợ chi tiết nhé!"
)

_SANITIZE_SUFFIX = (
    "\n\n_Lưu ý: Các thông tin trên mang tính tham khảo. "
    "Anh/chị vui lòng liên hệ trung tâm để xác nhận chi tiết._"
)


def validate(
    answer: str,
    *,
    tool_metadata: Optional[Dict[str, Any]] = None,
    tenant_id: Optional[str] = None,
    all_tenant_ids: Optional[List[str]] = None,
    route: Optional[str] = None,
) -> ValidationResult:
    """
    Validate an answer before returning to user.

    Returns ValidationResult with:
    - passed=True if answer is safe to return
    - violations: list of issues found
    - sanitized_answer: corrected answer if fixable
    - action: "pass", "sanitize", or "handoff"
    """
    if not answer or not answer.strip():
        return ValidationResult(passed=True, action="pass")

    all_violations: List[str] = []

    # 1) Internal data leak
    leak_violations = _check_internal_leak(answer)
    all_violations.extend(leak_violations)

    # 2) Numeric consistency
    num_violations = _check_numeric_consistency(answer, tool_metadata)
    all_violations.extend(num_violations)

    # 3) Tenant isolation
    tenant_violations = _check_tenant_isolation(answer, tenant_id, all_tenant_ids)
    all_violations.extend(tenant_violations)

    # 4) Forbidden promises
    promise_violations = _check_forbidden_promises(answer)
    all_violations.extend(promise_violations)

    if not all_violations:
        return ValidationResult(passed=True, action="pass")

    # Decide action based on severity
    has_leak = any("internal_leak" in v for v in all_violations)
    has_tenant_leak = any("cross_tenant_leak" in v for v in all_violations)
    has_numeric = any("numeric_mismatch" in v for v in all_violations)
    has_promise = any("promise:" in v for v in all_violations)

    logger.warning(
        "answer_validator violations=%d route=%s tenant=%s details=%s",
        len(all_violations),
        route or "-",
        tenant_id or "-",
        all_violations,
    )

    # Critical: internal leak or cross-tenant → handoff
    if has_leak or has_tenant_leak:
        return ValidationResult(
            passed=False,
            violations=all_violations,
            sanitized_answer=_HANDOFF_RESPONSE,
            action="handoff",
        )

    # Moderate: numeric mismatch → sanitize with disclaimer
    if has_numeric:
        return ValidationResult(
            passed=False,
            violations=all_violations,
            sanitized_answer=answer + _SANITIZE_SUFFIX,
            action="sanitize",
        )

    # Low: forbidden promises → sanitize with disclaimer
    if has_promise:
        return ValidationResult(
            passed=False,
            violations=all_violations,
            sanitized_answer=answer + _SANITIZE_SUFFIX,
            action="sanitize",
        )

    return ValidationResult(
        passed=False,
        violations=all_violations,
        sanitized_answer=answer + _SANITIZE_SUFFIX,
        action="sanitize",
    )
