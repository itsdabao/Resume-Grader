import re
import json
from typing import Dict, List, Optional, Any
from llama_index.core import VectorStoreIndex
from qdrant_client.http import models

from app.core.config import COLLECTION_NAME
from app.services.agentic.tools.base import ToolResult
from app.services.agentic.tools.utils import _sanitize_public_answer

def search_candidates_tool(
    skills: List[str],
    min_experience: int,
    query: str,
    index: VectorStoreIndex,
    limit: int = 5,
    stream: bool = False
) -> ToolResult:
    """
    Tìm kiếm ứng viên dựa trên kỹ năng (skills), số năm kinh nghiệm tối thiểu (min_experience) và câu truy vấn (query).
    """
    vector_store = index.vector_store
    client = vector_store.client

    must_conditions = []
    
    if min_experience > 0:
        must_conditions.append(
            models.FieldCondition(
                key="experience_years",
                range=models.Range(gte=min_experience)
            )
        )
        
    # Remove strict skills match, rely on vector similarity (semantic search) to find skills.
    # Because keyword match is case-sensitive and literal, which fails for "pytohn" or "AI Engineer".

    qdrant_filter = models.Filter(must=must_conditions) if must_conditions else None

    # We use vector search on the query if provided
    # Convert query to vector
    try:
        from llama_index.core import Settings
        query_embedding = Settings.embed_model.get_text_embedding(query)
    except Exception:
        query_embedding = [0.0] * 1024 # Fallback, might not work if dims mismatch
        
    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=limit * 3, # Fetch more to group by candidate
            with_payload=True
        )
        results = response.points
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Qdrant search failed: {e}")
        results = []

    # Group by candidate_id / source
    candidates = {}
    for res in results:
        payload = res.payload or {}
        src = payload.get("source") or payload.get("file_name") or "unknown"
        if src not in candidates:
            candidates[src] = {
                "score": res.score,
                "text": [],
                "skills": payload.get("skills", []),
                "experience_years": payload.get("experience_years", 0)
            }
        candidates[src]["text"].append(payload.get("text", ""))

    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)[:limit]
    
    if not sorted_candidates:
        ans = "Dạ em không tìm thấy ứng viên nào phù hợp với các tiêu chí này trong hệ thống."
        if stream:
            def _empty():
                yield {"type": "chunk", "content": ans}
                yield {"type": "result", "data": ToolResult(answer=ans, sources=[], metadata={"route": "hr_search"})}
            return _empty()
        return ToolResult(answer=ans, sources=[], metadata={"route": "hr_search"})

    # Prepare context for the LLM to synthesize
    context_chunks = []
    for src, data in sorted_candidates:
        exp = data["experience_years"]
        sks = ", ".join(data["skills"][:5])
        # Truncate text to avoid blowing up the context window
        full_text = "\n".join(data["text"])
        # Truncate text to avoid blowing up the context window
        full_text = "\n".join(data["text"])
        short_text = full_text[:400] + "..." if len(full_text) > 400 else full_text
        ctx = f"Ứng viên (File: {src}) - Kinh nghiệm: {exp} năm - Kỹ năng nổi bật: {sks}\nChi tiết: {short_text}"
        context_chunks.append(ctx)

    context_str = "\n\n---\n\n".join(context_chunks)
    
    # Synthesize with LLM - Optimized prompt for speed and quality
    prompt = (
        f"Bạn là trợ lý Nhân sự chuyên nghiệp. Hãy tóm tắt ngắn gọn điểm mạnh của các ứng viên sau đây dựa trên yêu cầu:\n"
        f"- Kỹ năng: {', '.join(skills) if skills else 'Tất cả'}\n"
        f"- Kinh nghiệm tối thiểu: {min_experience} năm\n\n"
        f"Danh sách ứng viên được tìm thấy:\n{context_str}\n\n"
        f"Hãy trả lời súc tích, so sánh và chỉ ra ứng viên tốt nhất. Viết bằng tiếng Việt."
    )
    
    try:
        from llama_index.core import Settings
        llm = Settings.llm
        if stream:
            def _stream_synth():
                yield {"type": "meta", "data": {"route": "hr_search", "sources": [s for s, _ in sorted_candidates]}}
                resp_gen = llm.stream_complete(prompt)
                full_txt = ""
                for chunk in resp_gen:
                    delta = getattr(chunk, "delta", "") or ""
                    if delta:
                        full_txt += delta
                        yield {"type": "chunk", "content": delta}
                
                yield {"type": "result", "data": ToolResult(
                    answer=_sanitize_public_answer(full_txt),
                    sources=[s for s, _ in sorted_candidates],
                    metadata={"route": "hr_search"}
                )}
            return _stream_synth()
        
        resp = llm.complete(prompt)
        ans = _sanitize_public_answer((resp.text or "").strip())
        return ToolResult(answer=ans, sources=[s for s, _ in sorted_candidates], metadata={"route": "hr_search"})
    except Exception as e:
        ans = f"Đã tìm thấy {len(sorted_candidates)} ứng viên, nhưng có lỗi khi tổng hợp: {e}"
        if stream:
            def _err():
                yield {"type": "chunk", "content": ans}
                yield {"type": "result", "data": ToolResult(answer=ans, sources=[s for s, _ in sorted_candidates], metadata={"route": "hr_search"})}
            return _err()
        return ToolResult(answer=ans, sources=[s for s, _ in sorted_candidates], metadata={"route": "hr_search"})
