from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class LlamaCppConfig:
    model_path: str
    n_ctx: int = 2048
    n_gpu_layers: int = 0
    n_threads: int = 0
    chat_format: str = "chatml"
    verbose: bool = False
    temperature: float = 0.2
    max_tokens: int = 1024


def _import_llamaindex_types():
    from llama_index.core.llms.custom import CustomLLM

    try:
        from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
    except Exception:
        from llama_index.core.llms.types import CompletionResponse, LLMMetadata  # type: ignore

    return CustomLLM, CompletionResponse, LLMMetadata


class LlamaCppLLM:  # placeholder when llama-index isn't installed
    def __init__(self, cfg: LlamaCppConfig):
        raise RuntimeError("llama-index is required to use LlamaCppLLM.")


def _build_llamaindex_llm_class():
    import threading

    CustomLLM, CompletionResponse, LLMMetadata = _import_llamaindex_types()

    class _LlamaCppLLM(CustomLLM):  # type: ignore[misc]
        """
        LlamaIndex-compatible LLM backed by `llama-cpp-python` (GGUF).

        This is a local in-process model. Calls are guarded by a lock because
        many llama.cpp backends are not re-entrant for concurrent requests.
        """

        model_path: str
        n_ctx: int = 2048
        n_gpu_layers: int = 0
        n_threads: int = 0
        chat_format: str = "chatml"
        verbose: bool = False
        temperature: float = 0.2
        max_tokens: int = 1024

        def __init__(self, cfg: LlamaCppConfig):
            super().__init__(
                model_path=cfg.model_path,
                n_ctx=int(cfg.n_ctx),
                n_gpu_layers=int(cfg.n_gpu_layers),
                n_threads=int(cfg.n_threads),
                chat_format=str(cfg.chat_format),
                verbose=bool(cfg.verbose),
                temperature=float(cfg.temperature),
                max_tokens=int(cfg.max_tokens),
            )
            try:
                from llama_cpp import Llama  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Missing dependency `llama-cpp-python`. Install it, or use LLM_PROVIDER=openai_compat + llama.cpp server."
                ) from e

            kwargs: dict[str, Any] = {
                "model_path": self.model_path,
                "n_ctx": int(self.n_ctx),
                "n_gpu_layers": int(self.n_gpu_layers),
                "verbose": bool(self.verbose),
            }
            if int(self.n_threads) > 0:
                kwargs["n_threads"] = int(self.n_threads)
            if str(self.chat_format).strip():
                kwargs["chat_format"] = str(self.chat_format).strip()

            self._llama = Llama(**kwargs)
            self._lock = threading.Lock()

        @property
        def metadata(self):
            return LLMMetadata(
                model_name=f"llama_cpp:{self.model_path}",
                context_window=int(self.n_ctx),
                num_output=int(self.max_tokens),
                is_chat_model=True,
            )

        def complete(self, prompt: str, **kwargs: Any):
            temp = float(kwargs.get("temperature", self.temperature))
            max_tokens = int(kwargs.get("max_tokens", self.max_tokens))

            with self._lock:
                resp = self._llama.create_chat_completion(
                    messages=[{"role": "user", "content": str(prompt)}],
                    temperature=temp,
                    max_tokens=max_tokens,
                )

            text: Optional[str] = None
            try:
                text = resp["choices"][0]["message"]["content"]
            except Exception:
                text = None
            return CompletionResponse(text=(text or "").strip(), raw=resp)

        def stream_complete(self, prompt: str, **kwargs: Any):
            yield self.complete(prompt, **kwargs)

    return _LlamaCppLLM


try:
    LlamaCppLLM = _build_llamaindex_llm_class()  # type: ignore[assignment]
except Exception:
    pass

