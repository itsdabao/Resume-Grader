import json
import os
import sys
from pathlib import Path

from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

from openai import OpenAI


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "app").exists() and (p / "data").exists():
            return p
        if (p / ".env.example").exists():
            return p
    # Fallback: evaluation/rag_eval/ -> repo root is usually 2 levels up
    return start.resolve().parents[2]


REPO_ROOT = _find_repo_root(Path(__file__).parent)
EVALS_DIR = Path(__file__).parent / "evals"


def _load_repo_dotenv() -> None:
    """
    Load repo-root `.env` without external dependencies (uv doesn't auto-load it).

    Supports:
    - KEY=VALUE
    - export KEY=VALUE
    - KEY: VALUE (best-effort)
    Strips surrounding single/double quotes.
    Does not override existing environment variables.
    """
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()

        key = None
        val = None
        if "=" in line:
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip()
        elif ":" in line:
            k, v = line.split(":", 1)
            key = k.strip()
            val = v.strip()

        if not key:
            continue
        if not key.replace("_", "").isalnum() or not (key[0].isalpha() or key[0] == "_"):
            continue
        if val is None:
            continue

        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]

        if key not in os.environ or not os.environ.get(key):
            os.environ[key] = val


def _get_groq_openai_compat_client() -> OpenAI:
    api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment.")
    base_url = (os.environ.get("GROQ_BASE_URL") or "").strip() or "https://api.groq.com/openai/v1"
    return OpenAI(api_key=api_key, base_url=base_url)


def _get_gemini_api_key() -> str:
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


def _get_gemini_model_name() -> str:
    return (os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash").strip()


def _get_groq_model_name() -> str:
    return (os.environ.get("GROQ_MODEL") or "openai/gpt-oss-120b").strip()


def _metric_llm_prefer_groq_fallback_gemini():
    """
    Auto preference:
    1) Groq (OpenAI-compatible) via GROQ_API_KEY/GROQ_BASE_URL
    2) Gemini via GEMINI_API_KEY (or GOOGLE_API_KEY)
    """
    # Prefer Groq
    try:
        groq_client = _get_groq_openai_compat_client()
        model = (os.environ.get("EVAL_MODEL") or _get_groq_model_name()).strip()
        return llm_factory(model, client=groq_client)
    except Exception:
        pass

    # Fallback to Gemini
    gemini_key = _get_gemini_api_key()
    if not gemini_key:
        raise RuntimeError("Missing GROQ_API_KEY and GEMINI_API_KEY/GOOGLE_API_KEY in environment.")
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Gemini fallback is enabled but `google-generativeai` is not installed.\n"
            "Install it with: `pip install google-generativeai`."
        ) from e

    genai.configure(api_key=gemini_key)
    client = genai.GenerativeModel(_get_gemini_model_name())
    return llm_factory(_get_gemini_model_name(), provider="google", client=client)


def _get_rag_client():
    # Make `evaluation/rag_eval` importable as a plain script.
    sys.path.insert(0, str(Path(__file__).parent))
    from rag import default_app_rag_client  # noqa: E402

    return default_app_rag_client(logdir=str(EVALS_DIR / "logs"))


_load_repo_dotenv()

EVALS_DIR.mkdir(parents=True, exist_ok=True)
(EVALS_DIR / "datasets").mkdir(parents=True, exist_ok=True)
(EVALS_DIR / "experiments").mkdir(parents=True, exist_ok=True)
(EVALS_DIR / "logs").mkdir(parents=True, exist_ok=True)

llm = _metric_llm_prefer_groq_fallback_gemini()


def load_dataset():
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir=str(EVALS_DIR),
    )

    rageval_jsonl = os.environ.get("RAGEVAL_JSONL_PATH") or str(
        REPO_ROOT / "evaluation" / "datasets" / "testset_vi_all_tenants.jsonl"
    )
    src_path = Path(rageval_jsonl)
    if not src_path.exists():
        raise FileNotFoundError(
            "Missing RAGEval-style source JSONL.\n"
            "Set RAGEVAL_JSONL_PATH or place it at:\n"
            f"  {src_path}"
        )

    limit = int((os.environ.get("RAG_EVAL_LIMIT") or "0").strip() or "0")
    kept = 0
    with src_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            tenant_id = (obj.get("tenant_id") or "").strip()
            query = obj.get("query") or {}
            question = (query.get("content") or "").strip()
            query_id = (query.get("query_id") or "").strip()
            query_type = (query.get("query_type") or "").strip()
            keypoints = (obj.get("ground_truth") or {}).get("keypoints") or []
            if not tenant_id or not question or not keypoints:
                continue

            grading_notes = "\n".join([f"- {kp}" for kp in keypoints if str(kp).strip()])
            if not grading_notes.strip():
                continue

            dataset.append(
                {
                    "tenant_id": tenant_id,
                    "query_id": query_id,
                    "query_type": query_type,
                    "question": question,
                    "grading_notes": grading_notes,
                }
            )
            kept += 1
            if limit and kept >= limit:
                break

    dataset.save()
    return dataset


my_metric = DiscreteMetric(
    name="correctness",
    prompt=(
        "Bạn là giám khảo đánh giá chất lượng câu trả lời của hệ thống RAG.\n"
        "Hãy kiểm tra xem câu trả lời có bao phủ các ý trong 'grading_notes' hay không.\n"
        "- Trả về 'pass' nếu câu trả lời bao phủ phần lớn các ý quan trọng.\n"
        "- Trả về 'fail' nếu thiếu nhiều ý quan trọng hoặc trả lời sai trọng tâm.\n"
        "Chỉ trả về đúng 1 từ: pass hoặc fail.\n\n"
        "Câu trả lời:\n{response}\n\n"
        "Các ý cần có (grading_notes):\n{grading_notes}\n"
    ),
    allowed_values=["pass", "fail"],
)


@experiment()
async def run_experiment(row):
    top_k = int((os.environ.get("RAG_TOP_K") or "5").strip() or "5")
    rag_client = _get_rag_client()
    response = rag_client.query(row["question"], tenant_id=row.get("tenant_id"), top_k=top_k)

    score = my_metric.score(
        llm=llm,
        response=response.get("answer", " "),
        grading_notes=row["grading_notes"],
    )

    experiment_view = {
        **row,
        "response": response.get("answer", ""),
        "score": score.value,
        "log_file": response.get("logs", " "),
        "top_k": top_k,
    }
    return experiment_view


async def main():
    dataset = load_dataset()
    print("dataset loaded successfully", dataset)
    experiment_results = await run_experiment.arun(dataset)
    print("Experiment completed successfully!")

    experiment_results.save()
    csv_path = EVALS_DIR / "experiments" / f"{experiment_results.name}.csv"
    print(f"\nExperiment results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

