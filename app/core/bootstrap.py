from __future__ import annotations

import os
import threading

# In some sandboxed/dev environments, network egress is blocked via a dummy proxy.
# Set offline mode early (before importing HuggingFace/transformers) to avoid long retry loops.
_proxy_vals = {
    (os.getenv("HTTP_PROXY") or "").strip().lower(),
    (os.getenv("HTTPS_PROXY") or "").strip().lower(),
    (os.getenv("ALL_PROXY") or "").strip().lower(),
}
_blocked_proxy = any(p in ("http://127.0.0.1:9", "https://127.0.0.1:9") for p in _proxy_vals)
if _blocked_proxy and not os.getenv("ALLOW_HF_DOWNLOAD"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from llama_index.core import Settings

from app.core.llama import init_llm_from_env, setup_embedding

_BOOTSTRAPPED_EMBEDDINGS = False
_BOOTSTRAPPED_RUNTIME = False
_LOCK = threading.RLock()


def bootstrap_embeddings_only() -> None:
    global _BOOTSTRAPPED_EMBEDDINGS
    if _BOOTSTRAPPED_EMBEDDINGS:
        return

    with _LOCK:
        # Double-check after acquiring lock (another thread may have finished)
        if _BOOTSTRAPPED_EMBEDDINGS:
            return
        try:
            setup_embedding()
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize embeddings. If you're offline/sandboxed, pre-download the HuggingFace model "
                f"or set EMBEDDING_MODEL_NAME to a local path. Original error: {e}"
            ) from e
        _BOOTSTRAPPED_EMBEDDINGS = True


def bootstrap_runtime() -> None:
    """
    Initialize LlamaIndex `Settings` for runtime Q&A:
    - embeddings
    - LLM (Gemini by default)
    Safe to call multiple times.
    """
    global _BOOTSTRAPPED_RUNTIME
    if _BOOTSTRAPPED_RUNTIME:
        return

    with _LOCK:
        if _BOOTSTRAPPED_RUNTIME:
            return
        bootstrap_embeddings_only()
        init_llm_from_env()
        _BOOTSTRAPPED_RUNTIME = True
