import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from llama_index.core import VectorStoreIndex

from app.core.config import RETRIEVAL_TOP_K
from app.services.agentic.evidence import extract_evidence_dict
from app.services.agentic.tools.base import ToolResult
from app.services.agentic.tools.utils import _sources_from_contexts
from app.services.rag.incontext_ralm import retrieve_hybrid_contexts


def _parse_compare_entities(query: str) -> List[str]:
    q = (query or "").strip()

    def clean_commands(s: str) -> str:
        s = re.sub(r"(?i)(?:^|\s)/(?:tenant|branch|index|llm|state|help|exit)\b(?:\s+\S+)?", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def norm(s: str) -> str:
        s = unicodedata.normalize("NFD", s or "")
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        s = s.replace("đ", "d").replace("Đ", "D").lower()
        s = re.sub(r"[^a-z0-9]+", " ", s).strip()
        return s

    q_clean = clean_commands(q)
    qn = norm(q_clean)

    if re.search(r"\bvs\b", qn):
        parts = re.split(r"\bvs\b", qn, maxsplit=1)
    elif "so sanh" in qn and re.search(r"\bvoi\b", qn):
        parts = re.split(r"\bvoi\b", qn, maxsplit=1)
    elif "so sanh" in qn and re.search(r"\bva\b", qn):
        parts = re.split(r"\bva\b", qn, maxsplit=1)
    else:
        return []

    cleaned = [p.strip(" -:\t") for p in parts if p and p.strip()]
    cleaned = [p.replace("so sanh", "").strip() for p in cleaned]
    cleaned = [p for p in cleaned if p]
    return cleaned[:2]


def comparison_tool(
    question: str,
    *,
    index: Optional[VectorStoreIndex],
    tenant_id: Optional[str] = None,
    branch_id: Optional[str] = None,
) -> ToolResult:
    entities = _parse_compare_entities(question)
    if len(entities) < 2:
        return ToolResult(
            answer="Dạ anh/chị muốn so sánh 2 khóa nào ạ? Ví dụ: “IELTS Foundation vs IELTS 6.5” hoặc “Giao tiếp vs TOEIC”.",
            sources=[],
            metadata={"route": "comparison"},
        )

    def _norm_for_match(s: str) -> str:
        s = unicodedata.normalize("NFD", s or "")
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        s = s.replace("đ", "d").replace("Đ", "D").lower()
        s = re.sub(r"[^a-z0-9]+", " ", s).strip()
        return s

    def _entity_mentioned(entity: str, blob: str) -> bool:
        ent = _norm_for_match(entity)
        txt = _norm_for_match(blob)
        if not ent or not txt:
            return False
        tokens = [t for t in ent.split() if len(t) >= 3]
        return any(re.search(rf"\b{re.escape(t)}\b", txt) for t in tokens)

    summaries: List[Tuple[str, Dict[str, object], List[Dict[str, object]], bool]] = []
    all_sources: List[str] = []
    for name in entities:
        q = f"{name} học phí thời lượng mục tiêu"
        if index is not None:
            retrieved = retrieve_hybrid_contexts(
                q,
                index,
                top_k_ctx=RETRIEVAL_TOP_K,
                tenant_id=tenant_id,
                branch_id=branch_id,
            )
            contexts = retrieved.get("contexts", []) or []
        else:
            from app.services.retrieval.bm25 import bm25_retrieve

            contexts = bm25_retrieve(q, top_k=RETRIEVAL_TOP_K, tenant_id=tenant_id, branch_id=branch_id)
        combined = "\n\n".join([str(c.get("text", "")) for c in contexts if isinstance(c, dict)])
        present = _entity_mentioned(name, combined)
        ev = extract_evidence_dict(combined) if present else {}
        summaries.append((name, ev, contexts, present))
        all_sources.extend(_sources_from_contexts(contexts))

    lines = ["Dạ em so sánh nhanh theo thông tin tìm được trong tài liệu:"]
    found: List[str] = []
    missing: List[str] = []
    for name, ev, _ctx, present in summaries:
        if not present:
            missing.append(name)
            continue
        tuition = ev.get("tuition_vnd", [])
        duration = ev.get("duration_months", [])
        ielts = ev.get("ielts_target", [])
        toeic = ev.get("toeic_target", [])
        cefr = ev.get("cefr_target", [])
        bits = []
        if tuition:
            vals = [int(x) for x in tuition if isinstance(x, (int, float))]
            if vals:
                mn, mx = min(vals), max(vals)
                bits.append((f"{mn:,}–{mx:,} VND" if mn != mx else f"{mn:,} VND").replace(",", "."))
        if duration:
            vals = [float(x) for x in duration if isinstance(x, (int, float))]
            if vals:
                mn, mx = min(vals), max(vals)
                bits.append((f"{mn:g}–{mx:g} tháng" if mn != mx else f"{mn:g} tháng"))
        if ielts:
            bits.append("IELTS " + ", ".join([str(x) for x in ielts]))
        if toeic:
            bits.append("TOEIC " + ", ".join([str(x) for x in toeic]))
        if cefr:
            bits.append("CEFR " + ", ".join([str(x) for x in cefr]))
        detail = "; ".join(bits) if bits else "chưa đủ dữ liệu rõ ràng (cần xác nhận thêm)"
        if bits:
            found.append(name)
            lines.append(f"- {name}: {detail}")
        else:
            missing.append(name)
    if not found:
        return ToolResult(
            answer="Dạ em chưa tìm thấy thông tin phù hợp cho cả 2 nội dung anh/chị muốn so sánh trong tài liệu/DB hiện có. Anh/chị cho em xin tên khóa cụ thể hoặc trung tâm/chi nhánh ạ.",
            sources=[],
            metadata={
                "route": "comparison",
                "retrieval": ("hybrid" if index is not None else "bm25_only"),
                "missing_entities": missing,
            },
        )

    if missing:
        lines.append(
            f"Dạ em tìm thấy thông tin cho {', '.join(found)}, còn {', '.join(missing)} hiện em chưa thấy trong tài liệu/DB."
        )

    lines.append("Anh/chị cho em biết mục tiêu và thời gian rảnh để em gợi ý khóa phù hợp nhất ạ.")

    dedup_sources = []
    seen = set()
    for s in all_sources:
        if s in seen:
            continue
        seen.add(s)
        dedup_sources.append(s)

    all_contexts: List[str] = []
    for _name, _ev, _ctxs, _ in summaries:
         all_contexts.extend([str(c.get("text", "")) for c in _ctxs])

    return ToolResult(
        answer="\n".join(lines),
        sources=dedup_sources,
        metadata={"route": "comparison", "retrieval": ("hybrid" if index is not None else "bm25_only")},
        context_texts=all_contexts
    )
