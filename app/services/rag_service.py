import logging
from typing import Dict, List, Optional

import qdrant_client
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app.core.config import (
    COLLECTION_NAME,
    EXAMPLES_TOP_K,
    FEWSHOT_PATH,
    QDRANT_HOST,
    QDRANT_PORT,
    RETRIEVAL_TOP_K,
)
from app.core.bootstrap import bootstrap_embeddings_only
from app.core.bootstrap import bootstrap_runtime
from app.core.llama import init_llm_from_env
from app.services.retrieval.vector_store import init_qdrant_collection
from app.services.agentic.service import agentic_query
from app.services.memory.service import memory_rag_query


logger = logging.getLogger(__name__)

_INDEX: Optional[VectorStoreIndex] = None


def build_index() -> VectorStoreIndex:
    """
    Khởi tạo embedding, kết nối Qdrant và load VectorStoreIndex dùng chung.
    Được cache vào biến toàn cục để tái sử dụng giữa các request.
    """
    global _INDEX
    if _INDEX is not None:
        return _INDEX

    # Ensure we never accidentally fall back to default embeddings (e.g., OpenAIEmbedding).
    # Note: Accessing `Settings.embed_model` may trigger lazy resolution in some llama-index versions.
    bootstrap_embeddings_only()

    # Ensure collection + payload indexes exist before loading the vector store
    client = init_qdrant_collection()
    logger.info("Qdrant ready at %s:%s", QDRANT_HOST, QDRANT_PORT)

    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    _INDEX = VectorStoreIndex.from_vector_store(vector_store)
    logger.info("Đã tải index từ collection: %s", COLLECTION_NAME)
    return _INDEX


def rag_query(
    question: str,
    *,
    tenant_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    channel: str = "cli",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    stream: bool = False,
) -> Dict[str, object]:
    """
    Hàm tiện ích RAG dùng chung cho CLI và backend.
    """
    # Make sure `.env` is loaded and `Settings.llm` is initialized even when startup skipped RAG init.
    bootstrap_runtime()
    index = build_index()
    if tenant_id and (session_id or user_id):
        return memory_rag_query(
            question,
            index=index,
            tenant_id=tenant_id,
            branch_id=branch_id,
            channel=channel,
            user_id=user_id,
            session_id=session_id,
            stream=stream,
        )
    return agentic_query(
        question,
        index=index,
        tenant_id=tenant_id,
        branch_id=branch_id,
        history=history or [],
        user_id=user_id,
        stream=stream,
    )
