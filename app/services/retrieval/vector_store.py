import types

import qdrant_client
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import VectorParams

from app.core.config import (
    BRANCH_FIELD,
    COLLECTION_NAME,
    ENABLE_BRANCH_FILTER,
    QDRANT_HOST,
    QDRANT_PORT,
    TENANT_FIELD,
    VECTOR_DISTANCE,
    VECTOR_SIZE,
)


def _ensure_qdrant_client_compat(client) -> None:
    """
    Ensure qdrant-client API compatibility for dependencies that expect
    `client.search(...)`.

    Some qdrant-client versions expose `search_points(...)` but not `search(...)`.
    LlamaIndex's Qdrant integration may call `client.search(...)`.
    """
    if hasattr(client, "search"):
        return

    def _search(self, collection_name, query_vector=None, query_filter=None, **kwargs):
        # Map common parameter names across versions.
        if "query_vector" in kwargs and query_vector is None:
            query_vector = kwargs.pop("query_vector")
        if "query_filter" in kwargs and query_filter is None:
            query_filter = kwargs.pop("query_filter")
        if "search_params" in kwargs and "params" not in kwargs:
            kwargs["params"] = kwargs.pop("search_params")

        # qdrant-client newer API uses `vector` + `filter`.
        if "vector" not in kwargs:
            kwargs["vector"] = query_vector
        if "filter" not in kwargs and query_filter is not None:
            kwargs["filter"] = query_filter

        if hasattr(self, "search_points"):
            return self.search_points(collection_name=collection_name, **kwargs)

        # Fallback to the generated HTTP client if present.
        http = getattr(self, "http", None)
        points_api = getattr(http, "points_api", None) if http is not None else None
        if points_api is not None and hasattr(points_api, "search_points"):
            try:
                from qdrant_client.http.models import SearchRequest

                req = SearchRequest(
                    vector=kwargs.get("vector"),
                    filter=kwargs.get("filter"),
                    limit=kwargs.get("limit"),
                    with_payload=kwargs.get("with_payload"),
                    with_vectors=kwargs.get("with_vectors"),
                    score_threshold=kwargs.get("score_threshold"),
                    params=kwargs.get("params"),
                )
                return points_api.search_points(collection_name=collection_name, search_request=req)
            except Exception:
                pass

        raise AttributeError("Qdrant client has neither `search` nor compatible `search_points` methods.")

    # Some qdrant-client builds may restrict setting new instance attributes;
    # patch class if needed.
    try:
        client.search = types.MethodType(_search, client)
    except Exception:
        try:
            setattr(client.__class__, "search", _search)
        except Exception:
            pass


def init_qdrant_collection():
    """
    Connect to Qdrant and ensure collection exists.
    If missing, create collection.
    If exists, keep existing data.
    """
    from app.core.config import PROJECT_ROOT
    qdrant_storage_path = str(PROJECT_ROOT / "data" / "qdrant_storage")
    
    if QDRANT_HOST in ("localhost", "127.0.0.1"):
        import os
        os.makedirs(qdrant_storage_path, exist_ok=True)
        print(f"Using embedded Qdrant (local disk mode) at: {qdrant_storage_path}")
        client = qdrant_client.QdrantClient(path=qdrant_storage_path)
    else:
        print(f"Connecting to remote Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
    _ensure_qdrant_client_compat(client)

    # Eager connectivity check so we fail fast with a clear message.
    try:
        _ = client.get_collections()
    except Exception as e:
        raise RuntimeError(
            f"Cannot connect to Qdrant.\n"
            f"Root error: {e}"
        ) from e
    print("Connected to Qdrant successfully.")

    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=VECTOR_DISTANCE,
            ),
        )
        print(f"Collection '{COLLECTION_NAME}' did not exist -> created successfully.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists -> keeping existing data.")

    # Create payload index for tenant_id / branch_id to enable fast per-tenant filter.
    tenant_payload_path = TENANT_FIELD
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=tenant_payload_path,
            field_schema="keyword",
        )
        print(f"Created payload index for '{tenant_payload_path}'.")
    except Exception:
        # Ignore if exists or server version does not support it.
        pass

    if ENABLE_BRANCH_FILTER:
        branch_payload_path = BRANCH_FIELD
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=branch_payload_path,
                field_schema="keyword",
            )
            print(f"Created payload index for '{branch_payload_path}'.")
        except Exception:
            pass

    # HR Agent Indexes
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="skills",
            field_schema="keyword",
        )
        print("Created payload index for 'skills'.")
    except Exception:
        pass

    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="experience_years",
            field_schema="integer",
        )
        print("Created payload index for 'experience_years'.")
    except Exception:
        pass

    return client


def get_storage_context(client):
    """Create StorageContext from Qdrant client for ingest/query use."""
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print("Storage context initialized successfully.")
    return storage_context
