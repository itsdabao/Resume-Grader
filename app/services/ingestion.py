from pathlib import Path
from typing import List, Optional
from llama_index.core import VectorStoreIndex
from app.services.documents import load_documents
from app.services.retrieval.vector_store import init_qdrant_collection, get_storage_context
from app.core.bootstrap import bootstrap_embeddings_only
from app.core.config import DATA_PATH
 
def run_ingestion(
    tenant_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    input_files: Optional[List[str]] = None,
    extra_metadata: Optional[dict] = None,
    *,
    pdf_engine: str = "auto",
    use_markdown_element_parser: bool = True,
    section_chunking: bool = True,
    section_heading_level: int = 2,
):
    import os, json
    from app.core.config import CHUNK_SIZE, CHUNK_OVERLAP, NODES_CACHE_PATH
    from app.services.ingestion_modern import IngestionOptions, load_documents_for_ingestion, build_nodes_for_ingestion

    print("Starting ingestion pipeline ...")

    # Ensure we never accidentally fall back to default embeddings (e.g., OpenAIEmbedding).
    # Note: Accessing `Settings.embed_model` may trigger lazy resolution in some llama-index versions.
    bootstrap_embeddings_only()
    client = init_qdrant_collection()
    try:
        storage_context = get_storage_context(client)

        opts = IngestionOptions(
            pdf_engine=pdf_engine,
            use_markdown_element_parser=use_markdown_element_parser,
            section_chunking=section_chunking,
            section_heading_level=section_heading_level,
        )
        documents = load_documents_for_ingestion(DATA_PATH, input_files=input_files, opts=opts)
        nodes = build_nodes_for_ingestion(
            documents,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            use_markdown_elements=use_markdown_element_parser,
            section_chunking=opts.section_chunking,
            section_heading_level=opts.section_heading_level,
        )

        # Attach tenant/branch metadata if provided
        if tenant_id or branch_id or extra_metadata:
            for n in nodes:
                try:
                    md = n.metadata
                    if not isinstance(md, dict):
                        md = {}
                except Exception:
                    md = {}
                if tenant_id:
                    md["tenant_id"] = tenant_id
                if branch_id:
                    md["branch_id"] = branch_id
                    
                if extra_metadata:
                    file_name = md.get("file_name")
                    if file_name:
                        resume_id = Path(file_name).stem
                        if resume_id in extra_metadata:
                            md.update(extra_metadata[resume_id])

                try:
                    n.metadata = md
                except Exception:
                    pass

        # Index nodes into Qdrant
        # Prefer direct constructor (nodes) and fall back to from_documents for compatibility.
        try:
            _ = VectorStoreIndex(nodes, storage_context=storage_context)
        except Exception:
            _ = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Persist nodes to JSONL for BM25 corpus
        cache_path = NODES_CACHE_PATH
        if tenant_id:
            base = Path(NODES_CACHE_PATH).parent
            if branch_id:
                cache_path = str(base / tenant_id / branch_id / "nodes.jsonl")
            else:
                cache_path = str(base / tenant_id / "nodes.jsonl")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            for n in nodes:
                node_id = None
                for attr in ("node_id", "id_", "id"):
                    try:
                        v = getattr(n, attr, None)
                    except Exception:
                        v = None
                    if isinstance(v, str) and v:
                        node_id = v
                        break
                try:
                    text = n.get_text()
                except Exception:
                    text = getattr(n, 'text', '')
                try:
                    md = n.metadata
                    if not isinstance(md, dict):
                        md = {}
                except Exception:
                    md = {}
                if tenant_id:
                    md["tenant_id"] = tenant_id
                if branch_id:
                    md["branch_id"] = branch_id
                
                if extra_metadata:
                    file_name = md.get("file_name")
                    if file_name:
                        resume_id = Path(file_name).stem
                        if resume_id in extra_metadata:
                            md.update(extra_metadata[resume_id])

                json.dump({'id': node_id, 'text': text, 'metadata': md}, f, ensure_ascii=False)
                f.write('\n')

        print("Done. Data has been written to Qdrant and nodes cached.")
    finally:
        if hasattr(client, "close"):
            client.close()


def _legacy_run_ingestion():
    print("🚀 Bắt đầu quá trình ingest dữ liệu...")

    setup_embedding()
    client = init_qdrant_collection()
    storage_context = get_storage_context(client)

    documents = load_documents(DATA_PATH)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    print("✅ HOÀN TẤT INGEST! Dữ liệu đã được ghi vào Qdrant.")
