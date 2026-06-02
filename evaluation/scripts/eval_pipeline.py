import os
import json
import logging
from pathlib import Path
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv

# Ragas & LlamaIndex
from llama_index.llms.groq import Groq
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LlamaIndexLLMWrapper
from datasets import Dataset

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# App imports
from app.services.rag_service import rag_query

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def load_testset_jsonl(path: str) -> List[Dict]:
    """Loads RAGEval-style JSONL testset."""
    data = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Testset not found at: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return data

def run_evaluation():
    # 1. Configuration
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.warning("GROQ_API_KEY not found. Please set it in .env.")

    # Judge: DeepSeek-R1 (via Groq) - Model: openai/gpt-oss-120b
    judge_llm = Groq(
        model="openai/gpt-oss-120b", 
        api_key=groq_api_key,
        temperature=0.0
    )
    ragas_judge = LlamaIndexLLMWrapper(judge_llm)

    # Embeddings for metrics (Answer Relevancy needs it)
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from ragas.embeddings import LlamaIndexEmbeddingsWrapper
    
    # Use Bge-m3 as used in the app
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    ragas_embed = LlamaIndexEmbeddingsWrapper(embed_model)

    # Assign Judge & Embeddings to Metrics
    faithfulness.llm = ragas_judge
    answer_relevancy.llm = ragas_judge
    answer_relevancy.embeddings = ragas_embed

    # 2. Load Data
    testset_path = os.getenv("RAGEVAL_JSONL_PATH") or str(REPO_ROOT / "evaluation" / "datasets" / "testset.jsonl")
    logger.info(f"Loading testset from {testset_path}...")
    try:
        raw_data = load_testset_jsonl(testset_path)
        logger.info(f"Loaded {len(raw_data)} samples.")
    except FileNotFoundError:
        logger.error("Testset not found. Please generate one under `evaluation/datasets/` first.")
        return

    # 3. Run RAG & Collect Results
    results_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    logger.info("Running RAG on testset...")
    for item in raw_data:
        # Parding RAGEval JSONL format
        # {"query": {"content": "..." }, "ground_truth": {"content": "..." }}
        try:
            q = item.get("query", {}).get("content")
            gt = item.get("ground_truth", {}).get("content")
            
            if not q:
                continue

            logger.info(f"Processing: {q}")
            
            # Call system under test
            response = rag_query(question=q) # response is Dict
            
            ans = str(response.get("answer", ""))
            ctxs = response.get("contexts", [])
            if isinstance(ctxs, list):
                ctxs = [str(c) for c in ctxs]
            else:
                ctxs = []

            results_data["question"].append(q)
            results_data["answer"].append(ans)
            results_data["contexts"].append(ctxs)
            results_data["ground_truth"].append(gt or "")

        except Exception as e:
            logger.error(f"Error processing question '{q}': {e}")
            continue

    # 4. Evaluate with RAGAS
    if not results_data["question"]:
        logger.error("No valid results to evaluate.")
        return

    dataset = Dataset.from_dict(results_data)
    
    logger.info("Starting evaluation with Judge (Groq/DeepSeek-R1)...")
    
    try:
        evaluation_results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_judge,
            raise_exceptions=False
        )
        
        # 5. Save Results
        df = evaluation_results.to_pandas()
        output_csv = r"d:\AI_Agent\data\evaluation_results.csv"
        df.to_csv(output_csv, index=False)
        logger.info(f"Evaluation complete. Results saved to {output_csv}")
        print(evaluation_results)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    run_evaluation()
