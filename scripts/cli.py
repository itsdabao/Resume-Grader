import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import pandas as pd


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("cli")


def ingest_global_knowledge(file_path: str):
    """Ingest a markdown file as global knowledge (tenant_id='global_public')."""
    from app.services.ingestion import run_ingestion

    print(f"Ingesting global knowledge from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    run_ingestion(tenant_id="global_public", input_files=[file_path])
    print("Ingestion complete.")


def import_faq_from_excel(excel_path: str, output_json_path: str):
    """Import FAQ from Excel (Col A: Question, Col B: Answer) and append/merge to JSON."""
    if not os.path.exists(excel_path):
        logger.error(f"Excel file not found: {excel_path}")
        return

    try:
        df = pd.read_excel(excel_path, header=None)
    except Exception as e:
        logger.error(f"Failed to read Excel: {e}")
        return

    new_items: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        q = str(row[0]).strip()
        a = str(row[1]).strip()
        q_norm = q.casefold()
        if not q or not a or q_norm in ("question", "câu hỏi"):
            continue
        item_id = f"faq_excel_{idx}"
        new_items.append({"id": item_id, "questions": [q], "answer": a})

    logger.info(f"Found {len(new_items)} FAQ items.")

    existing_data = []
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = []

    final_data = existing_data + new_items
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Successfully saved {len(final_data)} items to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="AI Agent Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_ingest = subparsers.add_parser("ingest-global", help="Ingest global knowledge file")
    parser_ingest.add_argument("--file", default="data/knowledge_base/general_concepts.md", help="Path to markdown file")

    parser_faq = subparsers.add_parser("import-faq", help="Import FAQ from Excel to Smalltalk JSON")
    parser_faq.add_argument("excel", help="Path to input Excel file (.xlsx)")
    parser_faq.add_argument("--output", default="app/resources/smalltalk_vi.json", help="Path to output JSON file")

    args = parser.parse_args()

    if args.command == "ingest-global":
        ingest_global_knowledge(args.file)
    elif args.command == "import-faq":
        import_faq_from_excel(args.excel, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
