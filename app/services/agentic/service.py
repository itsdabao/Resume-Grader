from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from llama_index.core import VectorStoreIndex

from app.core.config import ENABLE_SEMANTIC_ROUTER, FEWSHOT_PATH, TOXIC_MESSAGE
from app.services.guardrails import security_filter
from . import answer_validator, context_resolver, slot_manager

from .preprocess import preprocess_query
from .router import out_of_domain_answer, route_query
from .arguments import extract_comparison_args, extract_ticket_args, extract_tuition_calculator_args
from .tools import (
    ToolResult,
    create_ticket_tool,
    comparison_tool,
)


logger = logging.getLogger(__name__)


def _validate_and_return(
    out: Dict[str, object],
    *,
    tenant_id: Optional[str] = None,
) -> Dict[str, object]:
    """Sprint 3: validate answer before returning to user."""
    answer = str(out.get("answer") or "")
    tool_md = out.get("tool_metadata") if isinstance(out.get("tool_metadata"), dict) else None
    route = str(out.get("route") or "")
    v = answer_validator.validate(
        answer,
        tool_metadata=tool_md,
        tenant_id=tenant_id,
        route=route,
    )
    if v.passed:
        return out
    if v.sanitized_answer:
        out["answer"] = v.sanitized_answer
    if v.action == "handoff":
        out["route"] = f"{route}_validator_handoff"
    return out


def agentic_query(
    question: str,
    *,
    index: VectorStoreIndex,
    tenant_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    user_id: Optional[str] = None,
    stream: bool = False,
) -> Dict[str, object]:
    """
    CV Resume RAG agentic entrypoint: preprocessing -> semantic router -> RAG.
    """
    # --- Security Firewall ---
    sec = security_filter.check(question)
    if sec.blocked:
        logger.warning("security_filter blocked: category=%s reason=%s", sec.category, sec.reason)
        return {"answer": sec.safe_response or "", "sources": [], "route": f"security_{sec.category}"}

    p = preprocess_query(question)
    if p.toxic:
        return {"answer": TOXIC_MESSAGE, "sources": [], "route": "toxic"}

    decision = route_query(question)
    
    # Dynamically inject tenant_id if extracted by SLM
    if getattr(decision, "metadata_filter", None) and "tenant_id" in decision.metadata_filter:
        tenant_id = decision.metadata_filter["tenant_id"]
        
    logger.info(
        "router route=%s conf=%.2f reason=%s tenant=%s",
        decision.route,
        float(decision.confidence),
        decision.reason,
        tenant_id or "-",
    )

    active_route = decision.route

    if active_route == "general_chat":
        try:
            from llama_index.core import Settings
            if Settings.llm:
                prompt = (
                    "Bạn là trợ lý AI Nhân sự (HR Assistant). "
                    "Người dùng vừa gửi một câu hỏi bông đùa hoặc không liên quan đến tuyển dụng.\n"
                    "Hãy trả lời NGẮN GỌN (1-2 câu), vui vẻ, thân thiện, rồi nhẹ nhàng gợi nhắc "
                    "họ có thể hỏi về tìm kiếm CV, kỹ năng ứng viên.\n\n"
                    f'Câu hỏi: "{question}"\n'
                    "Trả lời:"
                )
                if stream:
                    def _stream_smalltalk():
                        yield {"type": "meta", "data": {"route": "general_chat", "sources": []}}
                        resp_gen = Settings.llm.stream_complete(prompt)
                        full_txt = ""
                        for chunk in resp_gen:
                            delta = getattr(chunk, "delta", "") or ""
                            if delta:
                                full_txt += delta
                                yield {"type": "chunk", "content": delta}
                        yield {"type": "result", "data": {"answer": full_txt, "sources": [], "route": "general_chat"}}
                    return _stream_smalltalk()
                else:
                    resp = Settings.llm.complete(prompt)
                    friendly = (resp.text or "").strip()
                    if friendly:
                        return {"answer": friendly, "sources": [], "route": "general_chat"}
        except Exception as e:
            logger.warning("smalltalk LLM fallback failed: %s", e)
        ans = "Dạ em là trợ lý Nhân sự. Anh/chị cần tìm ứng viên kỹ năng gì cứ nhắn em nhé!"
        if stream:
            def _fallback():
                yield {"type": "meta", "data": {"route": "general_chat", "sources": []}}
                yield {"type": "chunk", "content": ans}
                yield {"type": "result", "data": {"answer": ans, "sources": [], "route": "general_chat"}}
            return _fallback()
        return {"answer": ans, "sources": [], "route": "general_chat"}

    if active_route == "out_of_domain":
        ans = out_of_domain_answer()
        if stream:
            def _ood():
                yield {"type": "meta", "data": {"route": "out_of_domain", "sources": []}}
                yield {"type": "chunk", "content": ans}
                yield {"type": "result", "data": {"answer": ans, "sources": [], "route": "out_of_domain"}}
            return _ood()
        return {"answer": ans, "sources": [], "route": "out_of_domain"}

    active_route = "hr_search"

    # Regex-based extraction to avoid LLM latency
    skills = []
    min_exp = 0
    
    q_norm = " " + question.lower() + " "
    
    # 1. Extract min_experience: look for patterns like "3 năm", "5 nam", "2 year", "1 yrs"
    exp_matches = re.findall(r"(\d+)\s*(?:năm|nam|year|yr|yrs|tuổi|tuoi)", q_norm)
    if exp_matches:
        try:
            # Lấy số năm lớn nhất hoặc đầu tiên tìm thấy
            min_exp = int(exp_matches[0])
        except Exception:
            pass

    # 2. Extract skills based on a robust keyword dictionary
    common_skills = [
        "python", "java", "javascript", "react", "vue", "angular", "node", "typescript", 
        "ai", "machine learning", "nlp", "deep learning", "swift", "kotlin", "ios", "android",
        "c#", "c\\+\\+", "golang", "go ", "php", "ruby", "rust", "devops", "aws", "azure", 
        "docker", "kubernetes", "sql", "mysql", "postgresql", "nosql", "mongodb", "oracle",
        "banking", "credit", "risk", "finance", "tester", "qa", "qc", "scrum", "agile", 
        "pm", "project manager", "hr", "marketing", "sale", "telesale", "accounting", 
        "data analyst", "data engineer", "data scientist", "flutter", "react native", "figma",
        "ui", "ux", "design", "spring", "laravel", "django", "fastapi", "flask", "net"
    ]
    
    for skill in common_skills:
        # Sử dụng boundary check để tránh match "go" trong "google" hay "net" trong "network"
        pattern = rf"\b{re.escape(skill.strip())}\b"
        if re.search(pattern, q_norm):
            skills.append(skill.strip().title())

    # Nếu không match được skill cụ thể nào, trích xuất tất cả các từ viết hoa/tiếng Anh hoặc giữ nguyên để vector search xử lý
    # Vector search trong search_candidates_tool sẽ dựa trên full question nên skills chủ yếu để hiển thị/filter thêm.
    
    logger.info("Regex extraction result: skills=%s, min_experience=%d", skills, min_exp)

    from .tools import search_candidates_tool
    
    # We always call the search candidates tool
    if stream:
        gen = search_candidates_tool(
            skills=skills,
            min_experience=min_exp,
            query=question,
            index=index,
            limit=5,
            stream=True
        )
        
        def _stream_hr():
            for item in gen:
                if item["type"] == "meta":
                    item["data"]["route"] = active_route
                    yield item
                elif item["type"] == "result":
                    tr_obj = item["data"]
                    out = {"answer": tr_obj.answer, "sources": tr_obj.sources, "route": active_route}
                    if getattr(tr_obj, "metadata", None):
                        out["tool_metadata"] = tr_obj.metadata
                    out = _validate_and_return(out, tenant_id=tenant_id)
                    item["data"] = out
                    yield item
                else:
                    yield item
        return _stream_hr()

    tr = search_candidates_tool(
        skills=skills,
        min_experience=min_exp,
        query=question,
        index=index,
        limit=5,
        stream=False
    )
    
    out = {"answer": tr.answer, "sources": tr.sources, "route": active_route}
    if getattr(tr, "metadata", None):
        out["tool_metadata"] = tr.metadata
    return _validate_and_return(out, tenant_id=tenant_id)


def semantic_router_response(
    question: str,
    *,
    index: Optional[VectorStoreIndex] = None,
    tenant_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    user_id: Optional[str] = None,
) -> object:
    """
    Semantic-router output mode:
    - If a tool is needed: return a JSON-serializable dict {tool_name, arguments, thought}
    - Otherwise: return a plain text answer (string)
    """
    # --- Security Firewall (Sprint 1) ---
    sec = security_filter.check(question)
    if sec.blocked:
        logger.warning("security_filter blocked (semantic): category=%s reason=%s", sec.category, sec.reason)
        return sec.safe_response or ""

    p = preprocess_query(question)
    if p.toxic:
        return TOXIC_MESSAGE

    decision = route_query(p) if ENABLE_SEMANTIC_ROUTER else None
    route = decision.route if decision is not None else "course_search"

    if route == "smalltalk":
        return (decision.smalltalk_answer or "") if decision is not None else ""

    if route == "out_of_domain":
        if decision is not None and "language_mismatch" in decision.reason:
            return "Sorry I don't support this language. This chatbot is only supported in Vietnamese only."
        return out_of_domain_answer()

    if route == "create_ticket":
        args = extract_ticket_args(question)
        if not args.get("phone"):
            return "Dạ anh/chị cho em xin **SĐT** và **khung giờ thuận tiện** để tư vấn viên liên hệ hỗ trợ chi tiết nhé."
        if tenant_id:
            args["tenant_id"] = tenant_id
        if branch_id:
            args["branch_id"] = branch_id
        if user_id:
            args["user_id"] = user_id
        return {
            "tool_name": "create_ticket_tool",
            "arguments": args,
            "thought": "Người dùng muốn tư vấn/chuyển tư vấn viên, đã có SĐT nên tạo ticket để CSKH liên hệ.",
        }

    if route == "comparison":
        args = extract_comparison_args(question)
        if tenant_id:
            args["tenant_id"] = tenant_id
        if branch_id:
            args["branch_id"] = branch_id
        return {
            "tool_name": "comparison_tool",
            "arguments": args,
            "thought": "Câu hỏi yêu cầu so sánh giữa các khóa học, cần trích xuất tên khóa và tiêu chí.",
        }

    if route == "tuition_calculator":
        args = extract_tuition_calculator_args(question)
        if tenant_id:
            args["tenant_id"] = tenant_id
        if branch_id:
            args["branch_id"] = branch_id
        return {
            "tool_name": "tuition_calculator_tool",
            "arguments": args,
            "thought": "Câu hỏi về học phí/giảm giá/phụ phí, cần trích xuất biến tài chính để tính toán.",
        }

    if index is None:
        raise RuntimeError("semantic_router_response requires `index` for course_search answers.")

    tr = course_search_tool(
        question,
        index=index,
        fewshot_path=FEWSHOT_PATH,
        tenant_id=tenant_id,
        branch_id=branch_id,
        history=history or [],
    )
    return tr.answer
