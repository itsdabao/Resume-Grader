import os
import sys
import json
import logging
from pathlib import Path
import asyncio

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import CV_DATA_PATH
from app.core.llama import init_llm_from_env
from llama_index.core import Settings
import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

OUTPUT_FILE = Path(CV_DATA_PATH) / "ai_flags.json"

PROMPT_TEMPLATE = """Bạn là một chuyên gia tuyển dụng (HR) có kinh nghiệm, chuyên phát hiện các hồ sơ xin việc (CV) được viết bằng AI (như ChatGPT, Claude).
Dưới đây là một phần nội dung từ một CV. Hãy đọc kỹ và đánh giá xem CV này CÓ khả năng cao là do AI sinh ra hay không.

Các dấu hiệu của CV viết bằng AI:
1. Sử dụng quá nhiều từ ngữ khuôn sáo, to tát (ví dụ: spearhead, delve into, meticulously, symphony of...).
2. Cấu trúc câu quá hoàn hảo, thiếu tính cá nhân hoặc cảm xúc con người.
3. Các mô tả kinh nghiệm chung chung, liệt kê như sách giáo khoa mà không có chi tiết cụ thể về dự án thực tế.

Nội dung CV (Trích xuất):
\"\"\"
{text}
\"\"\"

Dựa vào các dấu hiệu trên, CV này có phải là do AI viết không?
TRẢ LỜI NGẮN GỌN BẰNG ĐÚNG 1 CHỮ: "TRUE" (nếu có dấu hiệu AI) hoặc "FALSE" (nếu giống người thật viết). Không giải thích thêm.
Trả lời:"""

def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        logging.error(f"Error reading {pdf_path}: {e}")
        return ""

def main():
    # Force the provider to be llama_cpp to use the local SLM
    os.environ["LLM_PROVIDER"] = "llama_cpp"
    init_llm_from_env()
    
    llm = Settings.llm
    if not llm:
        logging.error("Failed to initialize SLM (llama_cpp). Make sure LLAMA_CPP_MODEL_PATH is correct.")
        return

    # Load existing flags to support resuming
    ai_flags = {}
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                ai_flags = json.load(f)
            logging.info(f"Loaded {len(ai_flags)} existing flags.")
        except Exception as e:
            logging.error(f"Failed to load existing flags: {e}")

    base_path = Path(CV_DATA_PATH)
    categories = ["banking", "it"]
    pdf_files = []
    for cat in categories:
        cat_path = base_path / cat
        if cat_path.exists():
            pdf_files.extend(list(cat_path.glob("*.pdf")))

    logging.info(f"Found {len(pdf_files)} PDFs in total.")

    for i, pdf_path in enumerate(pdf_files):
        resume_id = pdf_path.stem
        if resume_id in ai_flags:
            continue
            
        logging.info(f"[{i+1}/{len(pdf_files)}] Analyzing {resume_id} for AI-generation...")
        text = extract_text_from_pdf(pdf_path)
        
        # Take the first ~2500 characters (around 500-600 words) to fit SLM context window (2048 tokens)
        sample_text = text[:2500] if len(text) > 2500 else text
        
        prompt = PROMPT_TEMPLATE.format(text=sample_text)
        
        try:
            resp = llm.complete(prompt)
            result_text = str(resp.text).strip().upper()
            
            # Simple parsing
            is_ai = "TRUE" in result_text
            
            ai_flags[resume_id] = is_ai
            if is_ai:
                logging.warning(f"  -> Flagged as AI-generated! (Response: {result_text})")
            else:
                logging.info(f"  -> Human-like. (Response: {result_text})")
                
            # Incremental save
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(ai_flags, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error calling SLM for {resume_id}: {e}")
            
    logging.info("Analysis complete! Results saved to ai_flags.json")

if __name__ == "__main__":
    main()
