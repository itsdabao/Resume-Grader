import sys
from pathlib import Path
import logging

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.ingestion import run_ingestion
from app.core.config import CV_DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def ingest_cvs():
    base_path = Path("d:/IMT_test/archive/data/data")
    
    categories = [
        ("BANKING", "banking"),
        ("INFORMATION-TECHNOLOGY", "it")
    ]
    
    for folder_name, tenant_id in categories:
        folder_path = base_path / folder_name
        if not folder_path.exists():
            logging.warning(f"Directory {folder_path} does not exist. Skipping.")
            continue
            
        pdf_files = list(folder_path.glob("*.pdf"))
        if not pdf_files:
            logging.warning(f"No PDFs found in {folder_path}. Skipping.")
            continue
            
        logging.info(f"Found {len(pdf_files)} PDFs for {tenant_id}. Starting ingestion...")
        
        # Load AI flags if available
        import json
        extra_meta = {}
        ai_flags_path = base_path / "ai_flags.json"
        if ai_flags_path.exists():
            try:
                with open(ai_flags_path, "r", encoding="utf-8") as f:
                    ai_flags = json.load(f)
                    for rid, is_ai in ai_flags.items():
                        extra_meta[rid] = {"is_ai_generated": is_ai}
            except Exception as e:
                logging.error(f"Error loading ai_flags.json: {e}")
                
        # Load extracted entities
        entities_path = REPO_ROOT / "data" / "cv_resumes" / "extracted_entities.json"
        if entities_path.exists():
            try:
                with open(entities_path, "r", encoding="utf-8") as f:
                    entities_list = json.load(f)
                    for ent in entities_list:
                        rid = ent.get("resume_id")
                        if not rid:
                            continue
                        if rid not in extra_meta:
                            extra_meta[rid] = {}
                        if ent.get("skills"):
                            extra_meta[rid]["skills"] = ent["skills"]
                        if ent.get("experience_years") is not None:
                            extra_meta[rid]["experience_years"] = ent["experience_years"]
            except Exception as e:
                logging.error(f"Error loading extracted_entities.json: {e}")

        # Ingest all files in the category folder
        input_files = [str(f) for f in pdf_files]
        run_ingestion(
            tenant_id=tenant_id,
            input_files=input_files,
            extra_metadata=extra_meta if extra_meta else None,
            pdf_engine="simple",  # We just need simple text extraction for BGE-M3
            use_markdown_element_parser=False,
            section_chunking=True, # Will use heuristics to split by headings
        )
        
        logging.info(f"Completed ingestion for {tenant_id}.")

if __name__ == "__main__":
    ingest_cvs()
