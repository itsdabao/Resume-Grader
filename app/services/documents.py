from pathlib import Path
from typing import List, Optional
from llama_index.core import SimpleDirectoryReader


def load_documents(data_path: str = "./data", input_files: Optional[List[str]] = None):
    """
    Đọc dữ liệu văn bản từ thư mục data_path hoặc danh sách file cụ thể
    và trả về danh sách document objects của LlamaIndex.

    - Khi input_files được truyền: chỉ đọc các file chỉ định.
    - Ngược lại: đọc toàn bộ thư mục (đệ quy) với các phần mở rộng phổ biến.
    """
    if input_files:
        files = [str(Path(fp)) for fp in input_files]
        print(f"Đang đọc dữ liệu từ danh sách file: {files}")
        reader = SimpleDirectoryReader(input_files=files)
        documents = reader.load_data()
    else:
        p = Path(data_path)
        if not p.exists():
            raise FileNotFoundError(f"Thư mục dữ liệu không tồn tại: {p}")

        print(f"Đang đọc dữ liệu từ: {p.resolve()} ...")

        # Các định dạng phổ biến cho RAG
        supported_exts = [".txt", ".md", ".pdf", ".docx", ".rtf"]
        reader = SimpleDirectoryReader(input_dir=str(p), recursive=True, required_exts=supported_exts)
        documents = reader.load_data()

    print(f"Đã đọc {len(documents)} tài liệu.")
    return documents
