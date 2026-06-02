from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

from llama_index.core import Settings

from .evidence import parse_money_to_vnd


FINANCIAL_EXTRACTION_SYSTEM_PROMPT = (
    "Bạn là chuyên gia thẩm định học phí. Hãy trích xuất dữ liệu tài chính từ đoạn văn bản được cung cấp theo các quy tắc nghiêm ngặt sau:\n"
    "\n"
    "Nhận diện Học phí gốc (Base Tuition): Chỉ lấy số tiền nằm ngay sau hoặc cùng dòng với tên khóa học. "
    "Định dạng thường là [Tên khóa học]: [Số tiền].\n"
    "\n"
    "Ví dụ: 'IELTS Speaking Intensive: 9.000.000' -> Base Tuition = 9,000,000.\n"
    "\n"
    "Phân biệt Phí phụ (Extra Fees): Các con số đi kèm với từ khóa 'Phí bảo lưu', 'Phí học bù', 'Phí tài liệu', 'Phí giáo trình' "
    "phải được đưa vào danh mục extra_fees, tuyệt đối không được coi là học phí chính.\n"
    "\n"
    "Cảnh báo độ tin cậy: Nếu tìm thấy nhiều con số nhưng không chắc chắn số nào thuộc về khóa học khách đang hỏi, "
    "hãy ưu tiên con số lớn nhất (thường là học phí).\n"
    "\n"
    "Định dạng đầu ra (JSON):\n"
    "```json\n"
    '{ "identified_course": "Tên khóa học", "base_tuition": 0, "extra_fees": 0, "confidence_score": 0.0 }\n'
    "```\n"
)


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
    if not raw.startswith("{"):
        m2 = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
        if m2:
            raw = m2.group(1).strip()
    return json.loads(raw)


def _safe_int(x: object) -> int:
    if isinstance(x, bool) or x is None:
        return 0
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return int(round(x))
    if isinstance(x, str):
        v = parse_money_to_vnd(x)
        if v is not None:
            return int(v)
        try:
            return int(re.sub(r"[^\d\-]", "", x))
        except Exception:
            return 0
    return 0


def _safe_float(x: object) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except Exception:
            return None
    return None


def refine_extracted_fees(llm_output: Dict[str, Any], course_name_query: str | None = None) -> Tuple[int, int, Dict[str, Any]]:
    """
    Safety heuristic to protect against "money noise" (e.g. 300k fees overriding 9m tuition).
    Returns (base_tuition_vnd, extra_fees_vnd, refined_payload).
    """
    MIN_TUITION_THRESHOLD = 1_000_000

    base = _safe_int(llm_output.get("base_tuition"))
    extra = _safe_int(llm_output.get("extra_fees"))

    if 0 < base < MIN_TUITION_THRESHOLD:
        extra = int(extra) + int(base)
        base = 0

    if base == 0 and extra >= MIN_TUITION_THRESHOLD:
        base = int(extra)
        extra = 0

    refined: Dict[str, Any] = dict(llm_output or {})
    refined["base_tuition_vnd"] = int(base)
    refined["extra_fees_vnd"] = int(extra)
    if course_name_query and not refined.get("identified_course"):
        refined["identified_course"] = str(course_name_query)
    return int(base), int(extra), refined


def extract_financials_with_llm(*, text: str, question: str) -> Dict[str, Any]:
    """
    LLM-backed extractor for finance fields from text. Returns a dict with:
    - identified_course
    - base_tuition_vnd
    - extra_fees_vnd
    - confidence_score
    - raw_text
    """
    llm = Settings.llm
    if llm is None:
        raise RuntimeError("LLM is not initialized (Settings.llm is None).")

    prompt = (
        f"{FINANCIAL_EXTRACTION_SYSTEM_PROMPT}\n"
        "VĂN BẢN:\n"
        f"{(text or '').strip()}\n\n"
        "CÂU HỎI/NGỮ CẢNH:\n"
        f"{(question or '').strip()}\n\n"
        "Chỉ trả về JSON hợp lệ theo schema."
    )
    resp = llm.complete(prompt)
    out_text = str(getattr(resp, "text", None) or resp)

    obj = _extract_json_object(out_text)
    base, extra, refined = refine_extracted_fees(obj, course_name_query=question)
    refined["confidence_score"] = _safe_float(obj.get("confidence_score"))
    refined["raw_text"] = out_text
    refined["base_tuition_vnd"] = int(base)
    refined["extra_fees_vnd"] = int(extra)
    return refined
