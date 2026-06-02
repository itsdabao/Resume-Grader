import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


logger = logging.getLogger(__name__)


def _ensure_agent_env() -> None:
    """
    Best-effort re-exec in conda env `agent` so llama-index + qdrant deps are present.
    """
    if os.getenv("RAGEVAL_BUILD_NO_REEXEC"):
        return
    try:
        import llama_index  # type: ignore  # noqa: F401
        import qdrant_client  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    cmd = [
        "conda",
        "run",
        "-n",
        "agent",
        "python",
        "-X",
        "utf8",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    env = dict(os.environ)
    env["RAGEVAL_BUILD_NO_REEXEC"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        r = subprocess.run(cmd, env=env)
        raise SystemExit(r.returncode)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Cannot find `conda` to re-exec into env `agent`.\n"
            "Run inside the correct env:\n"
            "  conda run -n agent python scripts/build_rageval_eval_vi.py"
        ) from e


def _load_env() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=str(REPO_ROOT / ".env"))
    except Exception:
        return


def _normalize_ws(s: str) -> str:
    return " ".join((s or "").replace("\r\n", "\n").replace("\r", "\n").split()).strip()


def _derive_tenant_id_from_md(md_path: Path) -> str:
    name = md_path.stem
    if name.lower().startswith("tenant_"):
        name = name[len("tenant_") :]
    name = re.sub(r"[^a-z0-9_]+", "", name.lower())
    return name or md_path.stem.lower()


def _iter_markdown_files(md_dir: Path, pattern: str = "tenant_*.md") -> List[Path]:
    return sorted([p for p in md_dir.glob(pattern) if p.is_file()])


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def _parse_json_list_from_model(text: str) -> List[Dict[str, Any]]:
    s = (text or "").strip()
    if s.startswith("```json"):
        s = s[len("```json") :].strip()
    if s.startswith("```"):
        s = s[len("```") :].strip()
    if s.endswith("```"):
        s = s[: -len("```")].strip()
    # Try direct parse
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    # Fallback: extract outermost JSON array
    a = s.find("[")
    b = s.rfind("]")
    if a != -1 and b != -1 and b > a:
        v = json.loads(s[a : b + 1])
        if isinstance(v, list):
            return v
    raise ValueError("Model did not return a JSON list.")


def _find_best_node_for_quote(quote: str, nodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    qn = _normalize_ws(quote)
    if not qn:
        return None
    for n in nodes:
        tn = _normalize_ws(str(n.get("text", "")))
        if qn and qn in tn:
            return n
    # Fallback: case-insensitive
    ql = qn.lower()
    for n in nodes:
        tl = _normalize_ws(str(n.get("text", ""))).lower()
        if ql and ql in tl:
            return n
    return None


def _fallback_node_by_overlap(answer: str, nodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ans = _normalize_ws(answer).lower()
    if not ans:
        return nodes[0] if nodes else None
    ans_tokens = set(re.findall(r"[a-zA-Z0-9À-ỹ]+", ans))
    if not ans_tokens:
        return nodes[0] if nodes else None
    best = None
    best_score = -1
    for n in nodes:
        txt = _normalize_ws(str(n.get("text", ""))).lower()
        toks = set(re.findall(r"[a-zA-Z0-9À-ỹ]+", txt))
        score = len(ans_tokens & toks)
        if score > best_score:
            best_score = score
            best = n
    return best


@dataclass(frozen=True)
class GeneratedQA:
    query_type: str
    question: str
    answer: str
    keypoints: List[str]
    evidence_quotes: List[str]


def _coerce_generated_qa(obj: Dict[str, Any]) -> GeneratedQA:
    q = obj.get("query", {}) or {}
    gt = obj.get("ground_truth", {}) or {}
    query_type = str(q.get("query_type", "") or "").strip() or "Factual Question"
    question = str(q.get("content", "") or "").strip()
    answer = str(gt.get("content", "") or "").strip()
    keypoints = gt.get("keypoints", []) or []
    if isinstance(keypoints, str):
        keypoints = [keypoints]
    keypoints = [str(x).strip() for x in keypoints if str(x).strip()]

    evidence_quotes = gt.get("evidence_quotes", []) or []
    if isinstance(evidence_quotes, str):
        evidence_quotes = [evidence_quotes]
    evidence_quotes = [str(x).strip() for x in evidence_quotes if str(x).strip()]

    if not question or not answer or not keypoints:
        raise ValueError("Invalid generated item: missing question/answer/keypoints.")
    return GeneratedQA(
        query_type=query_type,
        question=question,
        answer=answer,
        keypoints=keypoints,
        evidence_quotes=evidence_quotes,
    )


def _make_generator_llm():
    from llama_index.llms.groq import Groq  # type: ignore

    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in env/.env")
    model = (os.getenv("GROQ_MODEL") or "").strip() or "openai/gpt-oss-120b"
    return Groq(model=model, api_key=api_key, temperature=0.2)


def _prompt_generate_batch(*, tenant_id: str, markdown: str, n: int, seed: int) -> str:
    return (
        "Bạn là chuyên gia tạo bộ câu hỏi/đáp án để đánh giá hệ thống RAG cho một trung tâm Anh ngữ.\n"
        "Tài liệu nguồn (Markdown) thuộc 1 tenant duy nhất. Bạn chỉ được dùng thông tin trong tài liệu.\n\n"
        f"Tenant: {tenant_id}\n"
        f"Yêu cầu: Tạo đúng {n} mục QA bằng tiếng Việt. Không bịa thông tin ngoài tài liệu.\n"
        "Phân bổ độ khó: đa số dễ/trung bình, có một phần nâng cao.\n"
        "Các loại câu hỏi (query_type) chỉ dùng một trong các giá trị sau:\n"
        "- Factual Question\n"
        "- Policy/Procedure Question\n"
        "- Schedule Question\n"
        "- Comparison Question\n"
        "- Calculation Question\n"
        "- Summary Question\n"
        "- Multi-hop Question\n\n"
        "Mỗi mục phải trả về JSON object theo schema này:\n"
        "{\n"
        '  "query": {"query_type": "...", "content": "..."},\n'
        '  "ground_truth": {\n'
        '    "content": "...",\n'
        '    "keypoints": ["...","..."],\n'
        '    "evidence_quotes": ["..."]\n'
        "  }\n"
        "}\n\n"
        "Ràng buộc quan trọng:\n"
        "- evidence_quotes: 1-2 trích dẫn NGUYÊN VĂN, copy đúng từ tài liệu (giữ dấu tiếng Việt), mỗi quote <= 200 ký tự.\n"
        "- Các keypoints phải bám sát đáp án chuẩn và đủ để chấm completeness.\n"
        "- Không dùng markdown code block. Chỉ trả về 1 JSON array.\n\n"
        f"Seed: {seed}\n"
        "Tài liệu:\n"
        "--- START ---\n"
        f"{markdown}\n"
        "--- END ---\n"
    )


def generate_qa_for_tenant(
    *,
    tenant_id: str,
    markdown: str,
    total: int,
    batch_size: int,
    sleep_secs: float,
) -> List[GeneratedQA]:
    llm = _make_generator_llm()
    out: List[GeneratedQA] = []
    seed = 1
    while len(out) < total:
        need = min(batch_size, total - len(out))
        prompt = _prompt_generate_batch(tenant_id=tenant_id, markdown=markdown, n=need, seed=seed)
        seed += 1
        resp = llm.complete(prompt)
        raw = getattr(resp, "text", None) or str(resp)
        objs = _parse_json_list_from_model(raw)
        for o in objs:
            try:
                out.append(_coerce_generated_qa(o))
            except Exception as e:
                logger.warning("Skip invalid generated item for tenant=%s: %s", tenant_id, e)
        if sleep_secs:
            time.sleep(sleep_secs)
    return out[:total]


def ingest_markdown_for_tenant(*, tenant_id: str, md_path: Path) -> Path:
    from app.services.ingestion import run_ingestion

    run_ingestion(tenant_id=tenant_id, input_files=[str(md_path)])

    cache_path = REPO_ROOT / "data" / ".cache" / tenant_id / "nodes.jsonl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing ingestion cache nodes.jsonl for tenant={tenant_id}: {cache_path}")
    return cache_path


def copy_ingested_nodes_to_new_jsonl(*, tenant_id: str, cache_nodes_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"nodes_{tenant_id}.jsonl"
    shutil.copyfile(cache_nodes_path, out_path)
    return out_path


def build_eval_items_for_tenant(
    *,
    tenant_id: str,
    qas: List[GeneratedQA],
    nodes: List[Dict[str, Any]],
    domain: str,
    language: str,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from app.services.rag_service import rag_query

    eval_items: List[Dict[str, Any]] = []
    topk_logs: List[Dict[str, Any]] = []

    for i, qa in enumerate(qas, 1):
        query_id = f"{tenant_id}-{i:04d}"

        matched_nodes: List[Dict[str, Any]] = []
        for quote in qa.evidence_quotes[:2]:
            n = _find_best_node_for_quote(quote, nodes)
            if n and n not in matched_nodes:
                matched_nodes.append(n)
        if not matched_nodes:
            fb = _fallback_node_by_overlap(qa.answer, nodes)
            if fb:
                matched_nodes = [fb]

        gt_refs = [str(n.get("text", "")) for n in matched_nodes if str(n.get("text", "")).strip()]
        gt_doc_ids = [n.get("id") for n in matched_nodes if n.get("id") is not None]

        res = rag_query(qa.question, tenant_id=tenant_id)
        pred_content = str(res.get("answer", "") or "")
        contexts = res.get("contexts", []) or []
        if not isinstance(contexts, list):
            contexts = []
        pred_refs = [str(c) for c in contexts[:top_k]]

        topk_logs.append(
            {
                "tenant_id": tenant_id,
                "query_id": query_id,
                "query_type": qa.query_type,
                "question": qa.question,
                "answer": pred_content,
                "contexts_top_k": pred_refs,
                "sources": res.get("sources", []) or [],
                "route": res.get("route"),
            }
        )

        item = {
            "domain": domain,
            "language": language,
            "tenant_id": tenant_id,
            "query": {"query_id": query_id, "query_type": qa.query_type, "content": qa.question},
            "ground_truth": {
                "doc_ids": gt_doc_ids,
                "content": qa.answer,
                "references": gt_refs,
                "keypoints": qa.keypoints,
            },
            "prediction": {"content": pred_content, "references": pred_refs},
        }
        eval_items.append(item)

    return eval_items, topk_logs


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def init_logging() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )


def main() -> None:
    _ensure_agent_env()
    _load_env()
    init_logging()

    parser = argparse.ArgumentParser(description="Build a merged RAGEval evaluation JSONL for Vietnamese tenants.")
    parser.add_argument(
        "--md_dir",
        default=str(REPO_ROOT / "data" / "knowledge_base" / "preprocessed_markdown"),
        help="Folder containing tenant_*.md",
    )
    parser.add_argument(
        "--pattern",
        default="tenant_*.md",
        help="Glob pattern to select markdown files",
    )
    parser.add_argument(
        "--questions_per_tenant",
        type=int,
        default=50,
        help="Number of QA items to generate per tenant",
    )
    parser.add_argument("--batch_size", type=int, default=10, help="LLM generation batch size")
    parser.add_argument("--sleep_secs", type=float, default=1.0, help="Sleep between LLM batches")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K contexts to store in prediction.references")
    parser.add_argument("--domain", default="Education", help="domain field in output JSONL")
    parser.add_argument("--language", default="vi", help="language field in output JSONL")
    parser.add_argument(
        "--out_eval_jsonl",
        default=str(REPO_ROOT / "evaluation" / "datasets" / "testset_vi_all_tenants.jsonl"),
        help="Merged output JSONL path (RAGEval-style schema)",
    )
    parser.add_argument(
        "--out_topk_log_jsonl",
        default=str(REPO_ROOT / "data" / "knowledge_base" / "new_jsonl" / "topk_retrieval_log.jsonl"),
        help="Output JSONL path to store top-k retrieval logs",
    )
    parser.add_argument(
        "--out_nodes_dir",
        default=str(REPO_ROOT / "data" / "knowledge_base" / "new_jsonl" / "ingested"),
        help="Folder to copy ingested nodes.jsonl per tenant",
    )
    parser.add_argument(
        "--skip_ingest",
        action="store_true",
        help="Skip ingestion step (assumes Qdrant + data/.cache/<tenant>/nodes.jsonl already exist)",
    )
    args = parser.parse_args()

    md_dir = Path(args.md_dir)
    md_files = _iter_markdown_files(md_dir, pattern=args.pattern)
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in {md_dir} with pattern={args.pattern}")

    all_eval_items: List[Dict[str, Any]] = []
    all_topk_logs: List[Dict[str, Any]] = []

    for md_path in md_files:
        tenant_id = _derive_tenant_id_from_md(md_path)
        logger.info("Tenant=%s markdown=%s", tenant_id, md_path)

        cache_nodes_path: Optional[Path] = None
        if args.skip_ingest:
            cache_nodes_path = REPO_ROOT / "data" / ".cache" / tenant_id / "nodes.jsonl"
            if not cache_nodes_path.exists():
                raise FileNotFoundError(f"--skip_ingest set but missing {cache_nodes_path}")
        else:
            cache_nodes_path = ingest_markdown_for_tenant(tenant_id=tenant_id, md_path=md_path)

        # Copy nodes.jsonl into requested folder for traceability / offline use
        out_nodes_dir = Path(args.out_nodes_dir)
        copied_nodes_path = copy_ingested_nodes_to_new_jsonl(
            tenant_id=tenant_id,
            cache_nodes_path=cache_nodes_path,
            out_dir=out_nodes_dir,
        )
        nodes = _read_jsonl(copied_nodes_path)
        if not nodes:
            raise RuntimeError(f"No nodes loaded for tenant={tenant_id} from {copied_nodes_path}")

        markdown = md_path.read_text(encoding="utf-8")
        qas = generate_qa_for_tenant(
            tenant_id=tenant_id,
            markdown=markdown,
            total=int(args.questions_per_tenant),
            batch_size=int(args.batch_size),
            sleep_secs=float(args.sleep_secs),
        )

        eval_items, topk_logs = build_eval_items_for_tenant(
            tenant_id=tenant_id,
            qas=qas,
            nodes=nodes,
            domain=str(args.domain),
            language=str(args.language),
            top_k=int(args.top_k),
        )

        all_eval_items.extend(eval_items)
        all_topk_logs.extend(topk_logs)

    write_jsonl(Path(args.out_topk_log_jsonl), all_topk_logs)
    write_jsonl(Path(args.out_eval_jsonl), all_eval_items)
    logger.info("Wrote topk log: %s", args.out_topk_log_jsonl)
    logger.info("Wrote eval JSONL: %s", args.out_eval_jsonl)


if __name__ == "__main__":
    main()
