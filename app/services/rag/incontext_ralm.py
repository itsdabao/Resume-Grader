import json
import os
import time
import logging
from typing import List, Dict, Tuple

import numpy as np
from llama_index.core import Settings, VectorStoreIndex
from app.core.config import (
    USE_BM25,
    BM25_TOP_K,
    HYBRID_ALPHA,
    BM25_MAX_CHARS,
    DEBUG_VERBOSE,
    DEBUG_TOPN_PRINT,
    DEBUG_SHOW_PROMPT,
    MAX_PROMPT_CHARS,
    PER_CHUNK_PROMPT_MAX_CHARS,
    PROMPT_TOP_CONTEXTS,
    HISTORY_ENABLED,
    HISTORY_MAX_TURNS,
    HISTORY_MSG_MAX_CHARS,
    RERANK_USE_COSINE,
    RERANK_TOP_M,
    RERANK_WEIGHT,
    LLM_MAX_RETRIES,
    LLM_RETRY_INITIAL_DELAY,
    LLM_RETRY_BACKOFF,
    LLM_FALLBACK_TO_CONTEXT_ON_ERROR,
    LLM_FALLBACK_CONTEXT_SNIPPET_CHARS,
    SYSTEM_PROMPT_PATH,
    ENABLE_SMALLTALK,
    SMALLTALK_PATH,
    SMALLTALK_COSINE_THRESHOLD,
    ENABLE_DOMAIN_GUARD,
    DOMAIN_ANCHORS_PATH,
    DOMAIN_ANCHOR_COSINE_THRESHOLD,
    DOMAIN_KEYWORDS,
    DOMAIN_COSINE_THRESHOLD,
    OUT_OF_DOMAIN_MESSAGE,
    NO_MATCH_MESSAGE,
    BRANCH_FIELD,
    ENABLE_BRANCH_FILTER,
    ENFORCE_METADATA_FILTERS,
    REQUIRE_TENANT_ID,
    TENANT_FIELD,
)
from app.services.retrieval.bm25 import bm25_retrieve, bm25_retrieve_debug
from app.services.guardrails.domain_guard import DomainGuard
from app.services.guardrails.smalltalk import SmalltalkMatcher


logger = logging.getLogger(__name__)


def _embed(text: str) -> np.ndarray:
    model = Settings.embed_model
    if model is None:
        raise RuntimeError("Embed model is not initialized. Call setup_embedding() first.")
    # HuggingFaceEmbedding exposes get_text_embedding
    if hasattr(model, "get_text_embedding"):
        vec = model.get_text_embedding(text)
    elif hasattr(model, "embed"):
        vec = model.embed(text)
    else:
        raise RuntimeError("Embed model does not support text embedding.")
    return np.array(vec, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def _dbg_print(header: str):
    if DEBUG_VERBOSE:
        print(f"[DEBUG] {header}")


def _dbg_block(lines: List[str]):
    if DEBUG_VERBOSE:
        for ln in lines:
            print(f"[DEBUG] {ln}")


_SMALLTALK = SmalltalkMatcher(SMALLTALK_PATH)
_DOMAIN_GUARD = DomainGuard(DOMAIN_ANCHORS_PATH, keywords=DOMAIN_KEYWORDS)


def load_fewshot_examples(path: str) -> List[Dict[str, str]]:
    """Load few-shot Q/A examples from JSON file. Schema: [{"question": str, "answer": str}, ...]"""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out: List[Dict[str, str]] = []
        for item in data if isinstance(data, list) else []:
            q = item.get("question")
            a = item.get("answer")
            if isinstance(q, str) and isinstance(a, str):
                out.append({"question": q, "answer": a})
        return out
    except Exception:
        return []


def rank_examples_by_similarity(query: str, examples: List[Dict[str, str]], top_k: int) -> List[Dict[str, str]]:
    if not examples or top_k <= 0:
        return []
    q_vec = _embed(query)
    scored: List[Tuple[float, Dict[str, str]]] = []
    for ex in examples:
        try:
            s = _cosine(q_vec, _embed(ex["question"]))
            scored.append((s, ex))
        except Exception:
            continue
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:top_k]]


def _load_system_prompt(path: str) -> str:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if txt:
                    return txt
    except Exception:
        pass
    return (
        "Bạn là trợ lý RAG nói tiếng Việt. Trả lời ngắn gọn, rõ ràng, "
        "dựa trên ngữ cảnh được cung cấp; nếu không đủ thông tin thì nói không đủ dữ liệu; "
        "không bịa; luôn trích nguồn cuối câu trả lời."
    )


def _render_history(history: List[Dict[str, str]] | None) -> List[str]:
    if not HISTORY_ENABLED or not history:
        return []

    # Keep sticky/system context and rolling summary, then trim only chat turns.
    summary_lines: List[str] = []
    sticky_blocks: List[str] = []
    chat_turns: List[Dict[str, str]] = []
    for m in history:
        role = (m.get("role", "") or "").lower()
        content = (m.get("content", "") or "").strip()
        if not content:
            continue
        if role == "assistant" and content.strip().startswith("[Rolling Summary]"):
            summary_lines = ["Rolling Summary:", content.strip()[: (HISTORY_MSG_MAX_CHARS * 4)]]
        elif role == "system" or content.startswith("<user_context>") or content.startswith("[Long-term Memory]"):
            sticky_blocks.append(content[: (HISTORY_MSG_MAX_CHARS * 8)])
        else:
            if role in ("user", "assistant"):
                chat_turns.append({"role": role, "content": content})

    recent = chat_turns[-HISTORY_MAX_TURNS:]
    lines: List[str] = []
    if sticky_blocks:
        lines.append("User Context:")
        for blk in sticky_blocks:
            lines.append(blk)
        lines.append("")
    if summary_lines:
        lines.extend(summary_lines)
    if recent:
        if lines:
            lines.append("")
        lines.append("Conversation History:")
    for m in recent:
        role = m.get("role", "user").lower()
        content = (m.get("content", "") or "")[:HISTORY_MSG_MAX_CHARS]
        tag = "User" if role == "user" else "Assistant"
        lines.append(f"{tag}: {content}")
    return lines


def build_prompt(query: str, examples: List[Dict[str, str]], retrieved_texts: List[str], history: List[Dict[str, str]] | None = None) -> str:
    lines: List[str] = []
    # System instruction
    lines.append(_load_system_prompt(SYSTEM_PROMPT_PATH))
    # History (nếu có)
    hist_lines = _render_history(history)
    if hist_lines:
        lines.append("")
        lines.extend(hist_lines)
    if examples:
        lines.append("\nExamples:")
        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i} - Q: {ex['question']}")
            lines.append(f"Example {i} - A: {ex['answer']}")
    if retrieved_texts:
        lines.append("\nRetrieved Knowledge:")
        for i, t in enumerate(retrieved_texts, 1):
            lines.append(f"[Doc {i}] {t}")
    lines.append("\nUser Question:")
    lines.append(query)
    
    # Dynamic Output Control:
    # If conversation is ongoing, enforce brevity to avoid repetition.
    if history and len(history) > 2:
        lines.append(
            "\nLƯU Ý QUAN TRỌNG: "
            "Đây là hội thoại đang diễn ra. Hãy trả lời THẲNG VÀO VẤN ĐỀ (Direct Answer). "
            "KHÔNG giải thích lại các khái niệm/thông tin chung đã nhắc đến ở trên. "
            "Nếu thông tin không đổi, chỉ cần xác nhận ngắn gọn."
        )
    else:
        lines.append("\nAnswer clearly and concisely. If unsure, say you are unsure.")
    return "\n".join(lines)


def _extract_meta(n) -> Dict[str, str]:
    meta = {}
    try:
        meta = n.metadata
    except Exception:
        try:
            meta = n.node.metadata
        except Exception:
            meta = {}
    return meta if isinstance(meta, dict) else {}


def _extract_node_id(n) -> str | None:
    for attr in ("node_id", "id_", "id"):
        try:
            v = getattr(n, attr, None)
        except Exception:
            v = None
        if isinstance(v, str) and v:
            return v
    try:
        node = getattr(n, "node", None)
        if node is not None:
            for attr in ("node_id", "id_", "id"):
                try:
                    v = getattr(node, attr, None)
                except Exception:
                    v = None
                if isinstance(v, str) and v:
                    return v
    except Exception:
        pass
    return None


def _result_key(r: Dict[str, object]) -> str:
    rid = r.get("id")
    if isinstance(rid, str) and rid:
        return rid
    meta = r.get("meta") or {}
    if isinstance(meta, dict):
        # Prefer explicit IDs if present
        doc_id = meta.get("doc_id")
        chunk_id = meta.get("chunk_id")
        if isinstance(doc_id, str) and doc_id and isinstance(chunk_id, str) and chunk_id:
            return f"{doc_id}:{chunk_id}"
        for k in ("chunk_id", "doc_id"):
            v = meta.get(k)
            if isinstance(v, str) and v:
                return v
        # Otherwise include a stable source hint to reduce collisions.
        src = meta.get("source") or meta.get("file_name") or meta.get("file_path")
        if isinstance(src, str) and src:
            t = r.get("text") or ""
            if not isinstance(t, str):
                t = str(t)
            return f"{src}::{t[:120]}"
    t = r.get("text") or ""
    if not isinstance(t, str):
        t = str(t)
    return t[:200]


def _meta_allowed_for_scope(meta: Dict[str, object], *, tenant_id: str | None, branch_id: str | None) -> bool:
    """
    Fail-closed metadata gate applied after retrieval.
    This protects against vector-store/client variants that silently ignore filters.
    """
    m = meta if isinstance(meta, dict) else {}

    if tenant_id:
        allowed_tenants = {str(tenant_id), "global_public"}
        tv = m.get(TENANT_FIELD)
        if isinstance(tv, (list, tuple, set)):
            vals = {str(x) for x in tv}
            if not (vals & allowed_tenants):
                return False
        else:
            if str(tv) not in allowed_tenants:
                return False

    if ENABLE_BRANCH_FILTER and branch_id:
        bv = m.get(BRANCH_FIELD)
        if str(bv) != str(branch_id):
            return False

    return True


def _build_metadata_filters(tenant_id: str | None, branch_id: str | None):
    try:
        from llama_index.core.vector_stores.types import ExactMatchFilter, MetadataFilters, FilterOperator, MetadataFilter
    except Exception:
        return None
    
    filters_list = []
    
    # Global Knowledge Strategy:
    # If a tenant_id is provided, we want to retrieve docs that belong to THAT tenant
    # OR docs that belong to "global_public" (general knowledge).
    if tenant_id:
        # Check if we can use FilterOperator.IN or similar if LlamaIndex supports it.
        # Standard approach for "OR" in LlamaIndex MetadataFilters is tricky without 'condition=FilterCondition.OR'.
        # But we need AND(branch) AND (tenant==A OR tenant==global).
        
        # Simplified approach: Use "FilterOperator.IN" if supported for "tenant_id".
        # If not supported, we might need a custom Qdrant filter structure passed down, 
        # but LlamaIndex abstraction layer prefers MetadataFilters.
        
        # Try usage of IN operator (works with many vector stores including Qdrant in LlamaIndex).
        filters_list.append(
            MetadataFilter(
                key=TENANT_FIELD, 
                value=[tenant_id, "global_public"], 
                operator=FilterOperator.IN
            )
        )
    
    if ENABLE_BRANCH_FILTER and branch_id:
        filters_list.append(ExactMatchFilter(key=BRANCH_FIELD, value=branch_id))
        
    if not filters_list:
        return None
        
    return MetadataFilters(filters=filters_list)


def _vector_retrieve(
    index: VectorStoreIndex,
    user_query: str,
    top_k: int,
    *,
    tenant_id: str | None = None,
    branch_id: str | None = None,
) -> List[Dict[str, object]]:
    if REQUIRE_TENANT_ID and not tenant_id:
        return []

    metadata_filters = _build_metadata_filters(tenant_id, branch_id)

    # Try apply metadata filters at retriever layer (preferred).
    retriever = None
    applied_filters = False
    try:
        # Newer versions
        retriever = index.as_retriever(similarity_top_k=top_k, filters=metadata_filters)
        applied_filters = metadata_filters is None or True
    except Exception:
        retriever = index.as_retriever(similarity_top_k=top_k)
        if metadata_filters is not None:
            for attr in ("metadata_filters", "filters"):
                if hasattr(retriever, attr):
                    try:
                        setattr(retriever, attr, metadata_filters)
                        applied_filters = True
                        break
                    except Exception:
                        pass

    if ENFORCE_METADATA_FILTERS and metadata_filters is not None and not applied_filters:
        # Fail-closed: avoid cross-tenant leakage.
        return []

    nodes = retriever.retrieve(user_query)
    results: List[Dict[str, object]] = []
    strict_scope = bool(tenant_id) or bool(ENABLE_BRANCH_FILTER and branch_id)
    for n in nodes:
        node_id = _extract_node_id(n)
        meta = _extract_meta(n)
        if strict_scope and not _meta_allowed_for_scope(meta, tenant_id=tenant_id, branch_id=branch_id):
            continue
        try:
            txt = n.get_text()
        except Exception:
            txt = getattr(n, "text", "")
        score = 0.0
        try:
            score = float(getattr(n, "score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        results.append({"id": node_id, "text": (txt or "")[:1200], "score": score, "meta": meta})
    return results


def _hybrid_fuse(
    vec_res: List[Dict[str, object]],
    bm25_res: List[Dict[str, object]],
    *,
    final_top_k: int,
    rrf_k: int = 60,
) -> List[Dict[str, object]]:
    """
    Hybrid fusion using Reciprocal Rank Fusion (RRF).

    This avoids relying on incomparable score scales (vector similarity vs. BM25 score).
    HYBRID_ALPHA is used as a weight between the two rank contributions.
    """
    alpha = float(HYBRID_ALPHA)
    if not (0.0 <= alpha <= 1.0):
        alpha = 0.5

    fused: Dict[str, Dict[str, object]] = {}

    def _add_ranked(res: List[Dict[str, object]], *, weight: float, source: str) -> None:
        for rank, r in enumerate(res, 1):
            key = _result_key(r)
            add = float(weight) * (1.0 / float(rrf_k + rank))
            cur = fused.get(key)
            if cur is None:
                fused[key] = {
                    "id": r.get("id"),
                    "text": r.get("text", ""),
                    "meta": r.get("meta", {}) or {},
                    "score": add,
                }
            else:
                cur["score"] = float(cur.get("score", 0.0)) + add
                if not cur.get("id") and r.get("id"):
                    cur["id"] = r.get("id")
                # Merge meta without overwriting existing keys.
                m = r.get("meta") or {}
                if isinstance(m, dict):
                    cm = cur.get("meta") or {}
                    if not isinstance(cm, dict):
                        cm = {}
                    for k, v in m.items():
                        if k not in cm and v is not None:
                            cm[k] = v
                    cur["meta"] = cm

    _add_ranked(vec_res, weight=alpha, source="vector")
    _add_ranked(bm25_res, weight=(1.0 - alpha), source="bm25")

    fused_list = sorted(fused.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return fused_list[:final_top_k]


def query_with_incontext_ralm(
    user_query: str,
    index: VectorStoreIndex,
    *,
    fewshot_path: str,
    top_k_ctx: int = 5,
    top_k_examples: int = 3,
    tenant_id: str | None = None,
    branch_id: str | None = None,
    history: List[Dict[str, str]] | None = None,
    stream: bool = False,
):
    def _no_match() -> Dict[str, object]:
        return {"answer": NO_MATCH_MESSAGE, "sources": []}

    # 0) Smalltalk shortcut (avoid retrieval + LLM)
    if ENABLE_SMALLTALK:
        hit = _SMALLTALK.match(user_query, threshold=SMALLTALK_COSINE_THRESHOLD)
        if hit is not None:
            _dbg_block(
                [
                    f"Smalltalk hit: id={hit.id} score={hit.score:.3f}",
                    f"Matched: {hit.matched_question}",
                ]
            )
            return {"answer": hit.answer, "sources": []}

    # 1) Cheap in-domain pre-check (avoid retrieval + LLM for obvious out-of-domain)
    if ENABLE_DOMAIN_GUARD:
        decision = _DOMAIN_GUARD.decide(user_query, threshold=DOMAIN_ANCHOR_COSINE_THRESHOLD)
        _dbg_block(
            [
                f"Domain precheck: in_domain={decision.in_domain} score={decision.score:.3f} reason={decision.reason}",
                f"Matched anchor/kw: {decision.matched_anchor}",
            ]
        )
        if not decision.in_domain:
            return {"answer": OUT_OF_DOMAIN_MESSAGE, "sources": []}

    retrieval_metrics: Dict[str, object] | None = None

    # Retrieve knowledge: vector + optional BM25, then fuse
    _dbg_print("Starting retrieval pipeline")
    _dbg_block([
        f"Query: {user_query}",
        f"Vector top_k={top_k_ctx}, BM25 enabled={USE_BM25}, bm25_top_k={BM25_TOP_K}",
    ])
    t0 = time.perf_counter()
    t_vec0 = time.perf_counter()
    vec_res = _vector_retrieve(index, user_query, top_k_ctx, tenant_id=tenant_id, branch_id=branch_id)
    t_vec_ms = (time.perf_counter() - t_vec0) * 1000.0
    bm25_res: List[Dict[str, object]] = []
    bm25_stats = None
    t_bm25_ms = 0.0
    if USE_BM25:
        t_bm250 = time.perf_counter()
        bm25_dbg = bm25_retrieve_debug(
            user_query,
            top_k=BM25_TOP_K,
            max_chars=BM25_MAX_CHARS,
            tenant_id=tenant_id,
            branch_id=branch_id,
        )
        t_bm25_ms = (time.perf_counter() - t_bm250) * 1000.0
        bm25_tokens = bm25_dbg.get("q_tokens", [])
        bm25_res = bm25_dbg.get("results", [])
        bm25_stats = bm25_dbg.get("stats")
        _dbg_block(["BM25 tokens: " + str(bm25_tokens), "BM25 top results (score, src):"])
        for r in bm25_res[:DEBUG_TOPN_PRINT]:
            m = r.get("meta", {}) or {}
            src = m.get("file_name") or m.get("file_path") or "unknown"
            _dbg_block([f"  {r.get('score', 0.0):.4f} | {src}"])
    _dbg_block(["Vector top results (score, src):"])
    for r in vec_res[:DEBUG_TOPN_PRINT]:
        m = r.get("meta", {}) or {}
        src = m.get("file_name") or m.get("file_path") or "unknown"
        _dbg_block([f"  {r.get('score', 0.0):.4f} | {src}"])
    t_fuse0 = time.perf_counter()
    fused = _hybrid_fuse(vec_res, bm25_res, final_top_k=top_k_ctx)
    t_fuse_ms = (time.perf_counter() - t_fuse0) * 1000.0
    _dbg_block(["Fused results (RRF score, src):"])
    for r in fused[:DEBUG_TOPN_PRINT]:
        m = r.get("meta", {}) or {}
        src = m.get("file_name") or m.get("file_path") or "unknown"
        _dbg_block([f"  {r.get('score', 0.0):.4f} | {src}"])

    retrieval_metrics = {
        "len_q": len(user_query or ""),
        "tenant_id": tenant_id,
        "branch_id": branch_id,
        "alpha": float(HYBRID_ALPHA),
        "top_k_ctx": int(top_k_ctx),
        "bm25_enabled": bool(USE_BM25),
        "bm25_top_k": int(BM25_TOP_K),
        "n_vec": int(len(vec_res)),
        "n_bm25": int(len(bm25_res)),
        "n_fused": int(len(fused)),
        "t_vec_ms": round(float(t_vec_ms), 1),
        "t_bm25_ms": round(float(t_bm25_ms), 1),
        "t_fuse_ms": round(float(t_fuse_ms), 1),
        "t_total_retrieval_ms": round(float((time.perf_counter() - t0) * 1000.0), 1),
        "bm25": bm25_stats,
    }
    try:
        logger.info("retrieval_metrics %s", json.dumps(retrieval_metrics, ensure_ascii=False))
    except Exception:
        pass

    best_cosine: float | None = None

    # Optional rerank by cosine to query
    if RERANK_USE_COSINE and fused:
        try:
            q_vec = _embed(user_query)
            # take top M to re-score
            pool = fused[: min(RERANK_TOP_M, len(fused))]
            # normalize fused score
            max_f = max((float(it.get("score", 0.0)) for it in pool), default=0.0)
            rescored = []
            cos_scores = []
            for it in pool:
                cos = _cosine(q_vec, _embed(it["text"]))
                cos_scores.append(cos)
            if cos_scores:
                best_cosine = float(max(cos_scores))
            # min-max normalize cos
            if cos_scores:
                mn, mx = min(cos_scores), max(cos_scores)
                rng = (mx - mn) if mx > mn else 1.0
                cos_norm = [ (c - mn) / rng for c in cos_scores ]
            else:
                cos_norm = []
            for i, it in enumerate(pool):
                f_norm = (float(it.get("score", 0.0)) / max_f) if max_f > 0 else 0.0
                combined = (1.0 - RERANK_WEIGHT) * f_norm + RERANK_WEIGHT * (cos_norm[i] if i < len(cos_norm) else 0.0)
                rescored.append({**it, "score": combined})
            rescored.sort(key=lambda x: x["score"], reverse=True)
            fused = rescored[: top_k_ctx]
            _dbg_block(["After rerank (combined score, src):"])
            for r in fused[:DEBUG_TOPN_PRINT]:
                m = r.get("meta", {}) or {}
                src = m.get("file_name") or m.get("file_path") or "unknown"
                _dbg_block([f"  {r.get('score', 0.0):.4f} | {src}"])
        except Exception as e:
            _dbg_block([f"Rerank failed: {e}"])

    # In-domain / out-of-domain guardrail (avoid LLM on low-confidence retrieval)
    if ENABLE_DOMAIN_GUARD:
        # If rerank didn't run, compute cosine on the best available chunk (cheap-ish).
        if best_cosine is None and fused:
            try:
                q_vec = _embed(user_query)
                best_cosine = _cosine(q_vec, _embed(fused[0].get("text", "")))
            except Exception:
                best_cosine = None

        if fused and best_cosine is not None:
            _dbg_block([f"Domain guard: best_cosine={best_cosine:.3f} threshold={DOMAIN_COSINE_THRESHOLD:.3f}"])
            if float(best_cosine) < float(DOMAIN_COSINE_THRESHOLD):
                # Query is likely in-domain (passed the anchor/keyword precheck) but retrieval evidence is weak.
                # Prefer a "no matching info" message over "out of domain" to avoid confusing users.
                return _no_match()

    # Clamp exact top-N contexts for the prompt (post-rerank)
    selected = fused[: max(0, min(PROMPT_TOP_CONTEXTS, len(fused)))]

    # Apply prompt budget on selected contexts
    retrieved_texts: List[str] = []
    budget = MAX_PROMPT_CHARS
    overhead = 1200  # rough headroom for instructions + examples
    budget = max(1000, budget - overhead)
    used = 0
    for it in selected:
        t = it.get("text", "")
        if not t:
            continue
            
        # Append AI-generated warning if present in metadata
        meta = it.get("meta", {}) or {}
        if meta.get("is_ai_generated") is True:
            t = "[CẢNH BÁO: CV NÀY CÓ DẤU HIỆU ĐƯỢC VIẾT BẰNG AI (CHATGPT/CLAUDE), HÃY KIỂM TRA LẠI KỸ NĂNG THỰC TẾ]\n\n" + t
            
        t = t[:PER_CHUNK_PROMPT_MAX_CHARS]
        if used + len(t) > budget:
            break
        retrieved_texts.append(t)
        used += len(t)

    # Select few-shot examples
    examples_all = load_fewshot_examples(fewshot_path)
    examples = rank_examples_by_similarity(user_query, examples_all, top_k_examples)
    if examples:
        _dbg_block(["Few-shot selected (Q -> A):"])
        for i, ex in enumerate(examples[:DEBUG_TOPN_PRINT], 1):
            _dbg_block([f"  {i}. Q: {ex['question']}", f"     A: {ex['answer']}"])

    # Collect sources from fused items
    sources: List[str] = []
    for r in fused:
        m = r.get("meta", {}) or {}
        src = m.get("file_name") or m.get("file_path") or m.get("source") or "unknown"
        sources.append(str(src))

    if stream:
        def stream_generator():
            yield {"type": "meta", "data": {"sources": sources}}
            # Guardrail: if no context and no examples, avoid calling LLM
            if not retrieved_texts and not examples:
                ans = _no_match()["answer"]
                yield {"type": "chunk", "content": ans}
                out = {"answer": ans, "sources": sources, "contexts": [str(x.get("text", "")) for x in fused]}
                if retrieval_metrics is not None:
                    out["retrieval_metrics"] = retrieval_metrics
                yield {"type": "result", "data": out}
                return

            prompt = build_prompt(user_query, examples, retrieved_texts, history=history)
            if DEBUG_SHOW_PROMPT:
                head = prompt[:600].replace("\n", " ")
                _dbg_block([f"Prompt head(600): {head} ...", f"Prompt length: {len(prompt)} chars"])
            llm = Settings.llm
            if llm is None:
                raise RuntimeError("LLM is not initialized. Call init_llm_from_env() first.")
            delay = LLM_RETRY_INITIAL_DELAY
            for attempt in range(LLM_MAX_RETRIES):
                try:
                    resp_gen = llm.stream_complete(prompt)
                    full_text = ""
                    for chunk in resp_gen:
                        delta = getattr(chunk, "delta", "") or ""
                        if delta:
                            full_text += delta
                            yield {"type": "chunk", "content": delta}
                    out = {"answer": full_text, "sources": sources, "contexts": [str(x.get("text", "")) for x in fused]}
                    if retrieval_metrics is not None:
                        out["retrieval_metrics"] = retrieval_metrics
                    yield {"type": "result", "data": out}
                    return
                except Exception as e:
                    _dbg_block([f"LLM stream failed (attempt {attempt+1}/{LLM_MAX_RETRIES}): {e}"])
                    if attempt < LLM_MAX_RETRIES - 1:
                        time.sleep(delay)
                        delay *= LLM_RETRY_BACKOFF
                    else:
                        if LLM_FALLBACK_TO_CONTEXT_ON_ERROR:
                            msg = str(e)
                            msg_l = msg.lower()
                            is_quota = ("RESOURCE_EXHAUSTED" in msg) or ("429" in msg) or ("quota" in msg_l)
                            is_auth = ("401" in msg) or ("invalid_api_key" in msg_l) or ("api key" in msg_l) or ("apikey" in msg_l)
                            if is_quota or is_auth:
                                snippets = []
                                for t in retrieved_texts[:3]:
                                    if not t:
                                        continue
                                    snippets.append(t[:LLM_FALLBACK_CONTEXT_SNIPPET_CHARS])
                                if snippets:
                                    ans = (
                                        "Dạ hiện tại hệ thống đang tạm thời không gọi được LLM "
                                        + ("(API key không hợp lệ)." if is_auth else "(hết hạn mức/quota).")
                                        + " Em gửi anh/chị các đoạn thông tin liên quan nhất em tìm được trong tài liệu:\n\n"
                                        + "\n\n---\n\n".join(snippets)
                                    )
                                else:
                                    ans = OUT_OF_DOMAIN_MESSAGE if not is_auth else NO_MATCH_MESSAGE
                                yield {"type": "chunk", "content": ans}
                                out = {"answer": ans, "sources": sources, "contexts": [str(x.get("text", "")) for x in fused]}
                                if retrieval_metrics is not None:
                                    out["retrieval_metrics"] = retrieval_metrics
                                yield {"type": "result", "data": out}
                                return
                        raise e
        return stream_generator()

    # Build prompt and call LLM (Sync/Non-streaming)
    # Guardrail: if no context and no examples, avoid calling LLM
    if not retrieved_texts and not examples:
        return _no_match()

    prompt = build_prompt(user_query, examples, retrieved_texts, history=history)
    if DEBUG_SHOW_PROMPT:
        head = prompt[:600].replace("\n", " ")
        _dbg_block([f"Prompt head(600): {head} ...", f"Prompt length: {len(prompt)} chars"])
    llm = Settings.llm
    if llm is None:
        raise RuntimeError("LLM is not initialized. Call init_llm_from_env() first.")
    # LLM call with retry/backoff
    delay = LLM_RETRY_INITIAL_DELAY
    last_err = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = llm.complete(prompt)
            break
        except Exception as e:
            last_err = e
            _dbg_block([f"LLM call failed (attempt {attempt+1}/{LLM_MAX_RETRIES}): {e}"])
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= LLM_RETRY_BACKOFF
            else:
                if LLM_FALLBACK_TO_CONTEXT_ON_ERROR:
                    msg = str(e)
                    msg_l = msg.lower()
                    is_quota = ("RESOURCE_EXHAUSTED" in msg) or ("429" in msg) or ("quota" in msg_l)
                    is_auth = ("401" in msg) or ("invalid_api_key" in msg_l) or ("api key" in msg_l) or ("apikey" in msg_l)
                    # Provide a best-effort fallback without calling LLM again.
                    if is_quota or is_auth:
                        snippets = []
                        for t in retrieved_texts[:3]:
                            if not t:
                                continue
                            snippets.append(t[:LLM_FALLBACK_CONTEXT_SNIPPET_CHARS])
                        if snippets:
                            answer_text = (
                                "Dạ hiện tại hệ thống đang tạm thời không gọi được LLM "
                                + ("(API key không hợp lệ)." if is_auth else "(hết hạn mức/quota).")
                                + " Em gửi anh/chị các đoạn thông tin liên quan nhất em tìm được trong tài liệu:\n\n"
                                + "\n\n---\n\n".join(snippets)
                            )
                        else:
                            answer_text = OUT_OF_DOMAIN_MESSAGE if not is_auth else NO_MATCH_MESSAGE

                        out = {"answer": answer_text, "sources": sources}
                        if retrieval_metrics is not None:
                            out["retrieval_metrics"] = retrieval_metrics
                        return out
                raise
    answer_text = getattr(resp, "text", str(resp))

    out = {"answer": answer_text, "sources": sources, "contexts": [str(x.get("text", "")) for x in fused]}
    if retrieval_metrics is not None:
        out["retrieval_metrics"] = retrieval_metrics
    return out


def retrieve_hybrid_contexts(
    user_query: str,
    index: VectorStoreIndex,
    *,
    top_k_ctx: int = 5,
    tenant_id: str | None = None,
    branch_id: str | None = None,
) -> Dict[str, object]:
    """
    Retrieval-only API: returns fused contexts + metrics, without calling LLM.
    Intended for tool-first flows (calculator/comparison) to reduce LLM calls.
    """
    t0 = time.perf_counter()
    t_vec0 = time.perf_counter()
    vec_res = _vector_retrieve(index, user_query, top_k_ctx, tenant_id=tenant_id, branch_id=branch_id)
    t_vec_ms = (time.perf_counter() - t_vec0) * 1000.0

    bm25_res: List[Dict[str, object]] = []
    bm25_stats = None
    t_bm25_ms = 0.0
    if USE_BM25:
        t_bm250 = time.perf_counter()
        bm25_dbg = bm25_retrieve_debug(
            user_query,
            top_k=BM25_TOP_K,
            max_chars=BM25_MAX_CHARS,
            tenant_id=tenant_id,
            branch_id=branch_id,
        )
        t_bm25_ms = (time.perf_counter() - t_bm250) * 1000.0
        bm25_res = bm25_dbg.get("results", [])
        bm25_stats = bm25_dbg.get("stats")

    t_fuse0 = time.perf_counter()
    fused = _hybrid_fuse(vec_res, bm25_res, final_top_k=top_k_ctx)
    t_fuse_ms = (time.perf_counter() - t_fuse0) * 1000.0

    metrics = {
        "len_q": len(user_query or ""),
        "tenant_id": tenant_id,
        "branch_id": branch_id,
        "alpha": float(HYBRID_ALPHA),
        "top_k_ctx": int(top_k_ctx),
        "bm25_enabled": bool(USE_BM25),
        "bm25_top_k": int(BM25_TOP_K),
        "n_vec": int(len(vec_res)),
        "n_bm25": int(len(bm25_res)),
        "n_fused": int(len(fused)),
        "t_vec_ms": round(float(t_vec_ms), 1),
        "t_bm25_ms": round(float(t_bm25_ms), 1),
        "t_fuse_ms": round(float(t_fuse_ms), 1),
        "t_total_retrieval_ms": round(float((time.perf_counter() - t0) * 1000.0), 1),
        "bm25": bm25_stats,
    }

    return {"contexts": fused, "metrics": metrics}
