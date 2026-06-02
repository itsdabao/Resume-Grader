import argparse
import os

from llama_index.core import SimpleDirectoryReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick PDF text dump via LlamaIndex reader")
    parser.add_argument("file", help="Path to PDF file")
    args = parser.parse_args()

    pdf_path = args.file

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    try:
        reader = SimpleDirectoryReader(input_files=[pdf_path])
        documents = reader.load_data()

        print(f"Loaded {len(documents)} pages/documents.")
        for i, doc in enumerate(documents):
            print(f"--- Page {i+1} ---")
            print(doc.text)
            print("\n")
    except Exception as e:
        print(f"Error reading PDF: {e}")


if __name__ == "__main__":
    main()
