import re
from typing import Dict, List, Optional, Tuple

from llama_index.core import VectorStoreIndex

from app.core.config import RETRIEVAL_TOP_K
from app.services.agentic.evidence import parse_money_to_vnd
from app.services.rag.incontext_ralm import query_with_incontext_ralm
from app.services.agentic.tools.base import ToolResult
from app.services.agentic.tools.utils import (
    _norm_ascii_text,
    _sanitize_public_answer,
    _sources_from_contexts,
)
from app.services.rag.incontext_ralm import retrieve_hybrid_contexts


def _extract_course_catalog_from_texts(texts: List[str]) -> Dict[str, Dict[str, object]]:
    """
    Return map keyed by normalized course name:
    {key: {"name": display_name, "duration_months": float|None, "tuition_vnd": int|None}}
    """
    import unicodedata

    from app.services.agentic.tools.tuition import _extract_course_fee_pairs_from_text

    catalog: Dict[str, Dict[str, object]] = {}
    for txt in texts or []:
        block = str(txt or "")
        plain = unicodedata.normalize("NFD", block)
        plain = "".join(ch for ch in plain if unicodedata.category(ch) != "Mn")
        plain = plain.replace("đ", "d").replace("Đ", "D")
        for m in re.finditer(r"(?i)\b\d+\.\d+\.\s*([^\(\n]{3,80})\((\d+(?:[.,]\d+)?)\s*(?:thang)\b", plain):
            name_n = re.sub(r"\s+", " ", (m.group(1) or "")).strip()
            if not name_n:
                continue
            key = _norm_ascii_text(name_n)
            try:
                dur = float((m.group(2) or "").replace(",", "."))
            except Exception:
                dur = None
            item = catalog.get(key) or {"name": name_n.title(), "duration_months": None, "tuition_vnd": None}
            if isinstance(dur, float):
                item["duration_months"] = dur
            catalog[key] = item
        fee_map = _extract_course_fee_pairs_from_text(block)
        for fee_key, fee_val in fee_map.items():
            item = catalog.get(fee_key) or {"name": fee_key.title(), "duration_months": None, "tuition_vnd": None}
            item["tuition_vnd"] = int(fee_val)
            catalog[fee_key] = item
    return catalog


def course_search_tool(
    question: str,
    *,
    index: VectorStoreIndex,
    fewshot_path: str,
    tenant_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    stream: bool = False,
) -> ToolResult:
    q = str(question or "").strip()
    qn = _norm_ascii_text(q)
    ask_list_courses = any(
        k in qn
        for k in [
            "co nhung khoa hoc nao",
            "cac khoa hoc",
            "khoa hoc nao",
            "danh sach khoa hoc",
        ]
    )
    ask_recommend = any(
        k in qn
        for k in [
            "nen chon khoa nao",
            "goi y khoa",
            "tu van khoa",
        ]
    ) and ("ngan sach" in qn or "thang" in qn or "phat am" in qn)

    if ask_list_courses or ask_recommend:
        retrieval_query = "danh sach khoa hoc hoc phi" if ask_list_courses else q
        retrieved = retrieve_hybrid_contexts(
            retrieval_query,
            index,
            top_k_ctx=max(RETRIEVAL_TOP_K, 8),
            tenant_id=tenant_id,
            branch_id=branch_id,
        )
        contexts = retrieved.get("contexts", []) or []
        ctx_texts = [str(c.get("text", "")) for c in contexts if isinstance(c, dict)]
        catalog = _extract_course_catalog_from_texts(ctx_texts)
        if ask_list_courses and catalog:
            names = sorted(
                list(
                    {
                        str(v.get("name") or "").strip()
                        for v in catalog.values()
                        if str(v.get("name") or "").strip()
                    }
                )
            )
            if names:
                bullets = "\n".join([f"- {n}" for n in names[:10]])
                ans = "Dạ trung tâm hiện có các khóa chính:\n" + bullets
                if stream:
                    def _early_list():
                        src = _sources_from_contexts(contexts)
                        yield {"type": "meta", "data": {"sources": src}}
                        yield {"type": "chunk", "content": ans}
                        out = ToolResult(answer=_sanitize_public_answer(ans), sources=src, metadata={"route": "course_search", "heuristic": "course_list"}, context_texts=ctx_texts)
                        yield {"type": "result", "data": out}
                    return _early_list()
                return ToolResult(
                    answer=_sanitize_public_answer(ans),
                    sources=_sources_from_contexts(contexts),
                    metadata={"route": "course_search", "heuristic": "course_list"},
                    context_texts=ctx_texts,
                )

        if ask_recommend and catalog:
            budget = None
            for m in re.finditer(r"(?i)\b(\d+(?:[.,]\d+)?)\s*(tr|trieu|k|vnd|dong)\b", q):
                v = parse_money_to_vnd(m.group(0))
                if isinstance(v, int) and v > 0:
                    budget = int(v) if budget is None else max(int(budget), int(v))
            
            m_month = re.search(r"(?i)\b(\d+(?:[.,]\d+)?)\s*thang\b", qn) # Find month logic locally.
            target_month = None
            local_m_month = re.search(r"(?i)\b(\d+(?:[.,]\d+)?)\s*thang\b", qn) # Local to avoid changing the tool's behavior, since it parses 'qn' rather than 'q'.
            if local_m_month:
                 try:
                     target_month = float(local_m_month.group(1).replace(",", "."))
                 except Exception:
                     target_month = None

            wants_pron = ("phat am" in qn) or ("pronunciation" in qn)

            cands: List[Tuple[int, str, Dict[str, object]]] = []
            for item in catalog.values():
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                dur = item.get("duration_months")
                fee = item.get("tuition_vnd")
                if isinstance(target_month, (int, float)) and isinstance(dur, (int, float)):
                    if float(dur) > float(target_month) + 0.25:
                        continue
                if isinstance(budget, int) and isinstance(fee, int):
                    if int(fee) > int(budget):
                        continue
                score = 0
                if wants_pron and ("pronunciation" in _norm_ascii_text(name) or "phat am" in _norm_ascii_text(name)):
                    score += 3
                if isinstance(dur, (int, float)) and isinstance(target_month, (int, float)):
                    score += max(0, int(10 - abs(float(dur) - float(target_month)) * 3))
                if isinstance(fee, int) and isinstance(budget, int):
                    score += max(0, int((int(budget) - int(fee)) / 1_000_000))
                cands.append((score, name, item))

            cands.sort(key=lambda x: x[0], reverse=True)
            if cands:
                _score, name, item = cands[0]
                bits: List[str] = []
                if isinstance(item.get("duration_months"), (int, float)):
                    bits.append(f"{float(item['duration_months']):g} tháng")
                if isinstance(item.get("tuition_vnd"), int):
                    from app.services.agentic.tools.utils import _fmt_vnd_public
                    bits.append(_fmt_vnd_public(int(item["tuition_vnd"])))
                detail = ", ".join(bits)
                ans = (
                    f"Dạ theo nhu cầu của anh/chị, em gợi ý **{name}**"
                    + (f" ({detail})." if detail else ".")
                )
                if stream:
                    def _early_rec():
                        src = _sources_from_contexts(contexts)
                        yield {"type": "meta", "data": {"sources": src}}
                        yield {"type": "chunk", "content": ans}
                        out = ToolResult(answer=_sanitize_public_answer(ans), sources=src, metadata={"route": "course_search", "heuristic": "course_recommend"}, context_texts=ctx_texts)
                        yield {"type": "result", "data": out}
                    return _early_rec()
                return ToolResult(
                    answer=_sanitize_public_answer(ans),
                    sources=_sources_from_contexts(contexts),
                    metadata={"route": "course_search", "heuristic": "course_recommend"},
                    context_texts=ctx_texts,
                )

    result_or_gen = query_with_incontext_ralm(
        user_query=q,
        index=index,
        fewshot_path=fewshot_path,
        top_k_ctx=RETRIEVAL_TOP_K,
        top_k_examples=3,
        tenant_id=tenant_id,
        branch_id=branch_id,
        history=history or [],
        stream=stream,
    )
    
    if stream:
        def stream_wrapper():
            for item in result_or_gen:
                if item["type"] == "result":
                    res = item["data"]
                    item["data"] = ToolResult(
                        answer=_sanitize_public_answer(str(res.get("answer", ""))),
                        sources=[str(s) for s in (res.get("sources", []) or [])],
                        context_texts=[str(c) for c in (res.get("contexts", []) or [])],
                    )
                yield item
        return stream_wrapper()
        
    result = result_or_gen
    return ToolResult(
        answer=_sanitize_public_answer(str(result.get("answer", ""))),
        sources=[str(s) for s in (result.get("sources", []) or [])],
        context_texts=[str(c) for c in (result.get("contexts", []) or [])],
    )
