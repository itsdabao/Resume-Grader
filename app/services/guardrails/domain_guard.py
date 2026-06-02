from __future__ import annotations

import json
import logging
import os
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from llama_index.core import Settings

logger = logging.getLogger(__name__)


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


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
class DomainDecision:
    in_domain: bool
    score: float
    matched_anchor: Optional[str] = None
    reason: str = "anchor"


class DomainGuard:
    """
    Cheap in-domain/out-of-domain gate to avoid unnecessary retrieval/LLM calls.
    - Keyword short-circuit for common in-domain intents.
    - Otherwise semantic similarity to a small set of "domain anchors".
    """

    def __init__(self, anchors_path: str, *, keywords: Optional[List[str]] = None) -> None:
        self.anchors_path = anchors_path
        self.keywords = keywords or []
        self._loaded = False
        self._anchors: List[str] = []
        self._embs: List[Tuple[str, np.ndarray]] = []

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.anchors_path or not os.path.exists(self.anchors_path):
            self._anchors = []
            self._embs = []
            logger.debug("domain_guard: no anchors file at %s", self.anchors_path)
            return

        try:
            with open(self.anchors_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.debug("domain_guard: failed to load %s: %s", self.anchors_path, e)
            data = []

        anchors = [a for a in data if isinstance(a, str) and a.strip()] if isinstance(data, list) else []
        self._anchors = anchors
        self._embs = []
        for a in anchors:
            try:
                self._embs.append((a, _embed(a)))
            except Exception as e:
                logger.debug("domain_guard: embed failed for anchor=%s: %s", a, e)
                continue

    def decide(self, query: str, *, threshold: float) -> DomainDecision:
        query = (query or "").strip()
        if not query:
            return DomainDecision(in_domain=False, score=0.0, matched_anchor=None, reason="empty")

        # Keyword short-circuit (accent-insensitive)
        q_norm = _strip_accents(query).lower()
        for kw in self.keywords:
            if not kw:
                continue
            if _strip_accents(kw).lower() in q_norm:
                return DomainDecision(in_domain=True, score=1.0, matched_anchor=kw, reason="keyword")

        self._load()
        if not self._embs:
            return DomainDecision(in_domain=True, score=1.0, matched_anchor=None, reason="no_anchors")

        try:
            q_vec = _embed(query)
        except Exception:
            # If we can't embed, be permissive (avoid blocking real questions).
            return DomainDecision(in_domain=True, score=1.0, matched_anchor=None, reason="embed_failed")

        best_anchor = None
        best_score = -1.0
        for anchor, emb in self._embs:
            s = _cosine(q_vec, emb)
            if s > best_score:
                best_score = s
                best_anchor = anchor

        in_domain = float(best_score) >= float(threshold)
        return DomainDecision(in_domain=in_domain, score=float(best_score), matched_anchor=best_anchor, reason="anchor")
