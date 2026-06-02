import argparse
import os
import json
import logging
import uuid
from typing import List, Dict

from dotenv import load_dotenv
from llama_index.llms.groq import Groq

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()

def generate_testset(input_file: str, output_file: str, n_samples: int, model: str, temperature: float) -> None:
    """
    Generates a RAGEval-compliant dataset (JSONL) from the knowledge base
    using Groq (Llama 3 / DeepSeek).
    """
    # 1. Configuration
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY not found in env.")
            return

        llm = Groq(model=model, api_key=api_key, temperature=temperature)
    except Exception as e:
        logger.error(f"Failed to init Groq: {e}")
        return

    logger.info(f"Reading knowledge base from: {input_file}")
    if not os.path.exists(input_file):
        logger.error(f"File not found: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 2. Prompt for generation (RAGEval Schema)
    prompt = (
        "Bạn là người tạo bộ câu hỏi đánh giá cho hệ thống RAG.\n"
        "Hãy dựa hoàn toàn vào tài liệu dưới đây để tạo bộ QA.\n\n"
        "TÀI LIỆU:\n"
        "--- START ---\n"
        f"{content}\n"
        "--- END ---\n\n"
        f"Nhiệm vụ:\n"
        f"- Tạo {n_samples} cặp (question, answer) đa dạng từ dễ đến khó.\n"
        "- Mỗi câu trả lời phải có danh sách ý chính (keypoints) để chấm keypoint-matching.\n\n"
        "Yêu cầu format:\n"
        "- Output là JSON array (không dùng markdown code block).\n"
        "- Mỗi phần tử có schema:\n"
        "  {\n"
        "    \"domain\": \"Education\",\n"
        "    \"language\": \"vi\",\n"
        "    \"query\": {\"content\": \"...\", \"query_type\": \"Factual\"},\n"
        "    \"ground_truth\": {\"content\": \"...\", \"keypoints\": [\"...\", \"...\"]}\n"
        "  }\n"
        "- Chỉ dùng thông tin có trong tài liệu; không bịa.\n"
    )

    # 3. Generate
    logger.info(f"Generating testset with model={model} n={n_samples}...")
    try:
        response = llm.complete(prompt)
        response_text = response.text.strip()
        
        # Cleanup markdown if present
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "", 1)
        if response_text.startswith("```"):
             response_text = response_text.replace("```", "", 1)
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        # Parse list
        data_list = json.loads(response_text)
        
        logger.info(f"Generated {len(data_list)} items. Saving to JSONL...")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for item in data_list:
                # Enrich with IDs if missing
                if "query" in item:
                    if "query_id" not in item["query"]:
                        item["query"]["query_id"] = str(uuid.uuid4())
                
                # Write as single line JSON
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        logger.info(f"Saved RAGEval dataset to: {output_file}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON output: {e}\nResponse snippet: {response_text[:200]}")
    except Exception as e:
        logger.error(f"Generation failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Vietnamese RAGEval-style testset from a markdown KB.")
    parser.add_argument(
        "--input",
        default=os.getenv("KNOWLEDGE_BASE_MD") or "data/knowledge_base/general_concepts.md",
        help="Path to markdown knowledge base file",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("RAGEVAL_JSONL_PATH") or "evaluation/datasets/testset.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--n", type=int, default=int(os.getenv("RAG_EVAL_N", "50")), help="Number of samples")
    parser.add_argument(
        "--model",
        default=os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b",
        help="Groq model id (OpenAI-compatible)",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")

    args = parser.parse_args()
    generate_testset(args.input, args.output, args.n, args.model, args.temperature)
