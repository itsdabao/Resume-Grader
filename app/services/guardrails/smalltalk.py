from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from llama_index.core import Settings

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def _embed(text: str) -> np.ndarray:
    model = Settings.embed_model
    if model is None:
        raise RuntimeError("Embed model is not initialized. Call setup_embedding() first.")
    if hasattr(model, "get_text_embedding"):
        vec = model.get_text_embedding(text)
    elif hasattr(model, "embed"):
        vec = model.embed(text)
    else:
        raise RuntimeError("Embed model does not support text embedding.")
    return np.array(vec, dtype=np.float32)


@dataclass(frozen=True)
class SmalltalkHit:
    id: str
    answer: str
    score: float
    matched_question: str


class SmalltalkMatcher:
    """
    Semantic smalltalk router to avoid LLM calls for common chit-chat.
    Data format: JSON list of {id, questions: [..], answer}.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._loaded: bool = False
        self._items: List[Dict[str, object]] = []
        self._q_embeddings: List[Tuple[str, str, np.ndarray]] = []

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path or not os.path.exists(self.path):
            self._items = []
            self._q_embeddings = []
            logger.debug("smalltalk: no data file at %s", self.path)
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.debug("smalltalk: failed to load %s: %s", self.path, e)
            data = []

        items: List[Dict[str, object]] = []
        for obj in data if isinstance(data, list) else []:
            if not isinstance(obj, dict):
                continue
            item_id = obj.get("id")
            questions = obj.get("questions")
            answer = obj.get("answer")
            if not isinstance(item_id, str) or not isinstance(answer, str) or not isinstance(questions, list):
                continue
            qs = [q for q in questions if isinstance(q, str) and q.strip()]
            if not qs:
                continue
            items.append({"id": item_id, "questions": qs, "answer": answer})

        self._items = items
        self._q_embeddings = []
        # Precompute embeddings (small set) once per process.
        for it in self._items:
            for q in it["questions"]:
                try:
                    self._q_embeddings.append((str(it["id"]), q, _embed(q)))
                except Exception as e:
                    logger.debug("smalltalk: embed failed for item=%s: %s", it.get("id"), e)
                    continue

    def match(self, query: str, *, threshold: float) -> Optional[SmalltalkHit]:
        query = (query or "").strip()
        if not query:
            return None

        self._load()
        if not self._q_embeddings:
            return None

        try:
            q_vec = _embed(query)
        except Exception:
            return None

        best: Tuple[str, str, float] | None = None  # (id, matched_q, score)
        for item_id, q_text, q_emb in self._q_embeddings:
            s = _cosine(q_vec, q_emb)
            if best is None or s > best[2]:
                best = (item_id, q_text, s)

        if best is None or best[2] < float(threshold):
            return None

        item_id, matched_q, score = best
        for it in self._items:
            if it.get("id") == item_id:
                return SmalltalkHit(
                    id=str(item_id),
                    answer=str(it.get("answer", "")),
                    score=float(score),
                    matched_question=str(matched_q),
                )
        return None
