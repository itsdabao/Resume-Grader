import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "app").exists() and (p / "data").exists():
            return p
        if (p / ".env.example").exists():
            return p
    return start.resolve().parents[2]


REPO_ROOT = _find_repo_root(Path(__file__).parent)


@dataclass(frozen=True)
class ToolResult:
    answer: str
    sources: List[str]
    contexts: List[str]


class AppRAGClient:
    """
    Adapter to evaluate the *real* application RAG stack (Qdrant + tenant filters).

    Uses `app.services.rag.incontext_ralm.query_with_incontext_ralm` so we can:
    - enforce `tenant_id`/`branch_id` metadata filtering at retrieval layer
    - capture fused contexts for logging/inspection
    """

    def __init__(self, *, logdir: str = "evals/logs"):
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        self.logdir = logdir
        os.makedirs(self.logdir, exist_ok=True)

        from app.core.bootstrap import bootstrap_runtime
        from app.services.rag_service import build_index

        bootstrap_runtime()
        self.index = build_index()

    def query(
        self,
        question: str,
        *,
        tenant_id: Optional[str] = None,
        branch_id: Optional[str] = None,
        top_k: int = 5,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        from app.core.config import FEWSHOT_PATH
        from app.services.rag.incontext_ralm import query_with_incontext_ralm

        if run_id is None:
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(question) % 10000:04d}"

        result = query_with_incontext_ralm(
            question,
            self.index,
            fewshot_path=FEWSHOT_PATH,
            top_k_ctx=int(top_k),
            top_k_examples=3,
            tenant_id=tenant_id,
            branch_id=branch_id,
            history=[],
        )

        answer = str(result.get("answer", ""))
        contexts = [str(c) for c in (result.get("contexts") or [])][: int(top_k)]
        sources = [str(s) for s in (result.get("sources") or [])]

        log_payload = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "tenant_id": tenant_id,
            "branch_id": branch_id,
            "top_k": int(top_k),
            "question": question,
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "retrieval_metrics": result.get("retrieval_metrics"),
        }

        log_filename = f"app_rag_run_{run_id}_{datetime.now().isoformat().replace(':', '-').replace('.', '-')}.json"
        log_filepath = os.path.join(self.logdir, log_filename)
        with open(log_filepath, "w", encoding="utf-8") as f:
            json.dump(log_payload, f, ensure_ascii=False, indent=2)

        return {"answer": answer, "logs": log_filepath, "contexts": contexts, "sources": sources}


def default_app_rag_client(*, logdir: str = "evals/logs") -> AppRAGClient:
    return AppRAGClient(logdir=logdir)


if __name__ == "__main__":
    tenant_id = (os.environ.get("TENANT_ID") or "").strip() or None
    branch_id = (os.environ.get("BRANCH_ID") or "").strip() or None
    top_k = int((os.environ.get("TOP_K") or "5").strip() or "5")
    question = (os.environ.get("QUESTION") or "Học phí IELTS tại trung tâm là bao nhiêu?").strip()

    client = default_app_rag_client(logdir="logs")
    out = client.query(question, tenant_id=tenant_id, branch_id=branch_id, top_k=top_k)

    print("Answer:\n", out.get("answer", ""))
    print("\nLog:\n", out.get("logs", ""))

