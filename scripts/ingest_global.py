import os

from app.core.config import DATA_PATH
from app.services.ingestion import run_ingestion


def ingest_global() -> None:
    print("Ingesting Global Knowledge...")
    file_path = os.path.join(DATA_PATH, "general_concepts.md")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    run_ingestion(tenant_id="global_public", input_files=[file_path])
    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_global()

