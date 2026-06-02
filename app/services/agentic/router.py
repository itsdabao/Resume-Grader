"""
=============================================================================
 Agentic Router for CV Resume RAG
 ============================================================================
 
 Replaces the previous center-centric routing with CV-specific routing.
 Uses the Local SLM (Llama-3.2-3B / Qwen2.5-3B) loaded via llama_cpp.
 Routes:
 - cv_search: Find candidates, match skills, query experience
 - skills_analytics: Ask about skill distribution, general stats
 - general_chat: Smalltalk, off-topic, fallback
"""

import functools
import logging
import re
import json
from dataclasses import dataclass
from typing import Literal, Optional

logger = logging.getLogger(__name__)

RouteName = Literal[
    "cv_search",
    "skills_analytics",
    "general_chat",
    "out_of_domain",
]

@dataclass(frozen=True)
class RouteDecision:
    route: RouteName
    confidence: float
    reason: str
    metadata_filter: Optional[dict] = None

_VALID_INTENTS = {"cv_search", "skills_analytics", "general_chat"}

@functools.lru_cache(maxsize=256)
def _cached_llm_classify(query: str) -> Optional[tuple[str, Optional[dict]]]:
    """Classify intent via Local SLM. Returns (intent_name, metadata_filter)."""
    try:
        from llama_index.core import Settings
        if not Settings.llm:
            return None
            
        prompt = (
            "You are a routing assistant for an HR Resume search system.\n"
            "Classify the user query into one of the following intents:\n"
            "- cv_search: searching for candidates, finding resumes, asking about specific skills or experience.\n"
            "- skills_analytics: asking about overall statistics, distribution of skills, or comparing banking vs IT candidates as a whole.\n"
            "- general_chat: greeting, smalltalk, or completely off-topic questions.\n\n"
            "Also, if the user explicitly mentions 'banking' or 'IT' / 'information technology' / 'công nghệ thông tin', extract it into the category field. Otherwise leave it null.\n"
            "Must output valid JSON ONLY.\n\n"
            f'Query: "{query}"\n'
            'Return format:\n{"intent": "cv_search", "category": "BANKING"}'
        )
        
        resp = Settings.llm.complete(prompt)
        raw = (resp.text or "").strip()
        
        # Parse JSON
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            intent = data.get("intent", "general_chat").lower()
            category_val = data.get("category")
            
            metadata_filter = None
            if category_val:
                cat_upper = str(category_val).strip().upper()
                if "BANK" in cat_upper:
                    metadata_filter = {"tenant_id": "banking"}
                elif "IT" in cat_upper or "INFO" in cat_upper or "TECH" in cat_upper:
                    metadata_filter = {"tenant_id": "it"}
                    
            if intent in _VALID_INTENTS:
                return (intent, metadata_filter)
                
        # Regex fallback from LLM output
        if "cv_search" in raw.lower():
            return ("cv_search", None)
            
    except Exception as e:
        logger.warning("Local SLM router failed: %s", e)
    return None

def route_query(query: str) -> RouteDecision:
    q_lower = query.lower()
    
    # 1. Regex Pre-filters (Save Local Inference Time)
    smalltalk_patterns = [
        "xin chao", "chao ban", "hello", "hi ", "cam on", "bye", "ai day", "chao em",
        "ban la ai", "tro ly gi", "hey", "good morning", "good afternoon"
    ]
    if any(k in q_lower for k in smalltalk_patterns) and len(q_lower) < 30:
        return RouteDecision("general_chat", 0.9, "regex:smalltalk")
        
    analytics_patterns = [
        "thống kê", "phân bố", "so sánh it", "bao nhiêu ứng viên", "ti lệ", "ti le",
        "analytics", "bieu do", "biểu đồ", "phần trăm", "phan tram"
    ]
    if any(k in q_lower for k in analytics_patterns):
        return RouteDecision("skills_analytics", 0.8, "regex:analytics")

    # Mặc định rút tenant_id từ query bằng regex nếu có nhắc tới banking hay IT
    metadata = None
    if "bank" in q_lower or "ngân hàng" in q_lower or "ngan hang" in q_lower:
        metadata = {"tenant_id": "banking"}
    elif "it" in q_lower or "công nghệ" in q_lower or "cong nghe" in q_lower or "tech" in q_lower:
        metadata = {"tenant_id": "it"}

    # 2. Default Fallback thay vì gọi LLM
    return RouteDecision("cv_search", 0.9, "regex:default_rag", metadata_filter=metadata)

def out_of_domain_answer() -> str:
    return "Dạ em chỉ hỗ trợ tìm kiếm ứng viên và phân tích kỹ năng CV. Em không trả lời các câu hỏi ngoài lề được ạ."
