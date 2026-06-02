import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption


def process_pdf_to_markdown(*, file_path: str, output_dir: str) -> str:
    pipeline_options = PdfPipelineOptions()

    # Kích hoạt EasyOCR - nhận diện tiếng Việt
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = EasyOcrOptions()

    # Giữ cấu hình Table SOTA để lấy học phí chính xác
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.table_structure_options.do_cell_matching = False

    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"Input PDF not found: {src}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Đang bắt đầu xử lý SOTA (EasyOCR) cho: {src.name} ---")
    result = converter.convert(str(src))

    md_content = result.document.export_to_markdown()
    save_path = out_dir / (src.stem + ".md")
    save_path.write_text(md_content, encoding="utf-8")

    print(f"--- Thành công! File Markdown lưu tại: {save_path} ---")
    return str(save_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a PDF to Markdown using Docling + EasyOCR.")
    parser.add_argument("--input", required=True, help="Input PDF path")
    parser.add_argument("--output-dir", default=str(Path("data") / "knowledge_base" / "preprocessed_markdown"))
    args = parser.parse_args()

    process_pdf_to_markdown(file_path=args.input, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
