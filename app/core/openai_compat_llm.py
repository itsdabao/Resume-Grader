from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class OpenAICompatConfig:
    """
    Config for OpenAI-compatible Chat Completions endpoints (e.g., Groq).
    """

    api_key: str
    base_url: str
    model: str
    timeout_s: float = 60.0
    temperature: float = 0.2
    max_tokens: int = 1024
    context_window: int = 8192


def _import_llamaindex_types():
    # Keep imports lazy so this file can be parsed without llama-index installed.
    from llama_index.core.llms.custom import CustomLLM

    try:
        # Newer path
        from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
    except Exception:
        # Older path (best-effort)
        from llama_index.core.llms.types import CompletionResponse, LLMMetadata  # type: ignore

    return CustomLLM, CompletionResponse, LLMMetadata


class OpenAICompatLLM:  # will be replaced by a real LlamaIndex LLM subclass below
    def __init__(self, cfg: OpenAICompatConfig):
        raise RuntimeError("llama-index is required to use OpenAICompatLLM.")


def _build_llamaindex_llm_class():
    CustomLLM, CompletionResponse, LLMMetadata = _import_llamaindex_types()

    class _OpenAICompatLLM(CustomLLM):  # type: ignore[misc]
        """
        LlamaIndex-compatible LLM that calls OpenAI-style `/chat/completions`.

        We intentionally do NOT validate model names against OpenAI's catalog.
        This avoids errors like "Unknown model 'meta-llama/...'" when using Groq.
        """

        api_key: str
        base_url: str
        model: str
        timeout_s: float = 60.0
        temperature: float = 0.2
        max_tokens: int = 1024
        context_window: int = 8192

        def __init__(self, cfg: OpenAICompatConfig):
            super().__init__(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                model=cfg.model,
                timeout_s=cfg.timeout_s,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                context_window=cfg.context_window,
            )

        @property
        def metadata(self):
            # Mark as chat-capable; we still expose `.complete()` for this project.
            return LLMMetadata(
                model_name=self.model,
                context_window=int(self.context_window),
                num_output=int(self.max_tokens),
                is_chat_model=True,
            )

        def complete(self, prompt: str, **kwargs: Any):
            import httpx

            url = self.base_url.rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": float(kwargs.get("temperature", self.temperature)),
                "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            }
            # Optional OpenAI-compatible params (best-effort; ignored by providers that don't support them).
            if "response_format" in kwargs and kwargs["response_format"] is not None:
                payload["response_format"] = kwargs["response_format"]
            if "seed" in kwargs and kwargs["seed"] is not None:
                payload["seed"] = kwargs["seed"]

            with httpx.Client(timeout=float(self.timeout_s)) as client:
                resp = client.post(url, headers=headers, json=payload)

            if resp.status_code >= 400:
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                raise RuntimeError(f"OpenAI-compatible LLM error {resp.status_code}: {err}")

            data = resp.json()
            text: Optional[str] = None
            try:
                text = data["choices"][0]["message"]["content"]
            except Exception:
                text = None

            return CompletionResponse(text=(text or "").strip(), raw=data)

        def stream_complete(self, prompt: str, **kwargs: Any):
            # This project does not use streaming; implement minimal generator to satisfy CustomLLM.
            yield self.complete(prompt, **kwargs)

    return _OpenAICompatLLM


# Replace the placeholder with the real subclass at import time (when llama-index is installed).
try:
    OpenAICompatLLM = _build_llamaindex_llm_class()  # type: ignore[assignment]
except Exception:
    # Keep placeholder error for environments without llama-index.
    pass
