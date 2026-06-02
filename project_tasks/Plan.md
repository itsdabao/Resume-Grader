# Hệ thống CV Resume RAG — Kế hoạch Triển khai (v3: Tối ưu Hardware & Production)

## 1. Chiến lược Kiến trúc: Thực thi Kết hợp Cloud-Edge
Do giới hạn khắt khe của máy tính phát triển (**tối đa 4GB VRAM**), kiến trúc được chia nhỏ để tránh lỗi tràn bộ nhớ (OOM) trong khi vẫn giữ được tốc độ và độ thông minh cao:

| Phân hệ | Mô hình Thực thi | Model Phân công | Tiêu thụ VRAM | Lý do |
|---|---|---|---|---|
| **Xử lý Nặng (Trích xuất Offline)** | Cloud APIs | Chính: **Groq** (nhanh)<br>Dự phòng: **Gemini** (ngữ cảnh) | ~0GB | Việc xử lý hàng loạt 235 CV thành file JSON có cấu trúc đòi hỏi tốc độ và context window lớn vượt quá 4GB VRAM. |
| **Tác vụ Thời gian thực (Agentic Router)** | Edge (Local SLM) | **Qwen2.5-1.5B** hoặc **Llama-3.2-3B-Instruct** (Q4_K_M GGUF) qua Ollama/llama.cpp | ~2-3GB | Chạy nhanh, độ trễ thấp, dùng để phân loại ý định người dùng & điều hướng từ khóa cho câu hỏi đầu vào. |
| **Tạo Embedding** | Edge (Local) | **BAAI/bge-m3** (1024-d) | <1GB | Siêu nhẹ, có thể chạy hiệu quả trên CPU hoặc GPU mức tối thiểu để phục vụ tìm kiếm. |
| **Dự phòng OCR (Bonus 10%)** | Edge (CPU) | **Tesseract + OpenCV** | ~0GB | Chỉ kích hoạt nếu chữ trích xuất < 50 ký tự, giữ cho GPU rảnh rỗi. |

## 2. Luồng Dữ liệu & Kiến trúc Mới

1. **Phase 1: Pipeline Dữ liệu An toàn (Edge + Cloud)**
   - Giải nén 235 PDFs → Check độ dài chữ (> 50) → pdfplumber. Nếu không → OCR.
   - Pydantic Validation & Sandbox → Cloud LLM (Groq) → `extracted_entities.json`.

2. **Phase 2: RAG Indexing (Local Edge)**
   - Đưa raw text vào `ingestion_modern.py` → BGE-M3 Embedding → Qdrant Vector Store (kèm metadata).

3. **Phase 3: Pipeline Truy vấn Hybrid**
   - User Query → Local SLM Router.
   - Nếu `cv_search` → Hybrid Retrieval (Vector + BM25) → Cosine Rerank.
   - Kết quả trả về → Cloud LLM (Groq) tổng hợp câu trả lời cuối.

## 3. Lớp Bảo mật & An toàn
- **Chống Prompt Injection (Sandbox):** Nhốt payload CV vào thẻ `<PAYLOAD>`.
- **Ngăn chặn Hallucination (Pydantic):** Ép trường thông tin thiếu thành `None`/`null`.
