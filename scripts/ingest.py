import sys
import argparse
import logging
from pathlib import Path
import re

# Đảm bảo import được gói src/* khi chạy từ thư mục scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.ingestion import run_ingestion
from app.core.config import DATA_PATH
from app.core.bootstrap import bootstrap_embeddings_only

logging.basicConfig(
    filename="logs/ingest.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def _derive_tenant_from_filename(path: Path) -> str:
    name = path.stem
    # Nếu tên bắt đầu bằng tenant_, lấy phần sau; ngược lại dùng nguyên tên
    if name.lower().startswith("tenant_"):
        name = name[len("tenant_") :]
    # Chuẩn hoá: chữ thường, chỉ giữ a-z0-9 và _
    safe = re.sub(r"[^a-z0-9_]+", "", name.lower())
    return safe or name.lower()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest pipeline")
    parser.add_argument("--tenant", default=None, help="Tenant ID (gắn vào metadata)")
    parser.add_argument("--branch", default=None, help="Branch ID (tuỳ chọn, gắn vào metadata)")
    parser.add_argument("--file", default=None, help="Đường dẫn file đơn lẻ để ingest")
    parser.add_argument(
        "--pdf-engine",
        choices=["auto", "llamaparse", "simple"],
        default="auto",
        help="Cách đọc PDF: auto (có key thì LlamaParse), llamaparse, hoặc simple",
    )
    parser.add_argument(
        "--no-md-elements",
        action="store_true",
        help="Tắt MarkdownElementNodeParser (fallback SentenceSplitter)",
    )
    parser.add_argument(
        "--no-section-chunking",
        action="store_true",
        help="Tắt chunk theo mục lớn (heading) khi ingest markdown/llamaparse",
    )
    parser.add_argument(
        "--section-heading-level",
        type=int,
        default=2,
        help="Heading level để tách mục lớn (vd: 2 cho '## ...')",
    )
    parser.add_argument(
        "--auto-from-filenames",
        action="store_true",
        help="Quét DATA_PATH và ingest từng file, suy ra tenant từ tên file (tenant_*.pdf)",
    )
    args = parser.parse_args()

    try:
        bootstrap_embeddings_only()
        if args.auto_from_filenames:
            base = Path(DATA_PATH)
            if not base.exists():
                raise FileNotFoundError(f"Thư mục dữ liệu không tồn tại: {base}")
            # Chỉ ingest các định dạng phổ biến
            exts = {".pdf", ".md", ".txt", ".docx", ".rtf"}
            files = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts]
            if not files:
                print("Không tìm thấy file nào để ingest.")
            else:
                print(f"Tìm thấy {len(files)} file để ingest trong {base.resolve()}.")
            for fp in files:
                tenant = _derive_tenant_from_filename(fp)
                print(f"\n[Auto] Ingest tenant={tenant} file={fp}")
                run_ingestion(
                    tenant_id=tenant,
                    branch_id=args.branch,
                    input_files=[str(fp)],
                    pdf_engine=args.pdf_engine,
                    use_markdown_element_parser=not args.no_md_elements,
                    section_chunking=not args.no_section_chunking,
                    section_heading_level=args.section_heading_level,
                )
            print("\nIngest hàng loạt hoàn tất.")
        else:
            if args.file:
                run_ingestion(
                    tenant_id=args.tenant,
                    branch_id=args.branch,
                    input_files=[args.file],
                    pdf_engine=args.pdf_engine,
                    use_markdown_element_parser=not args.no_md_elements,
                    section_chunking=not args.no_section_chunking,
                    section_heading_level=args.section_heading_level,
                )
            else:
                run_ingestion(
                    tenant_id=args.tenant,
                    branch_id=args.branch,
                    pdf_engine=args.pdf_engine,
                    use_markdown_element_parser=not args.no_md_elements,
                    section_chunking=not args.no_section_chunking,
                    section_heading_level=args.section_heading_level,
                )
            print("Ingest hoàn tất.")
    except Exception as e:
        logging.error(f"Lỗi khi ingest: {e}")
        print(f"Lỗi: {e}")
