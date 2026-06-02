# Phase 1: Data Prep & Extraction (Xử lý Dữ liệu và Trích xuất)

## Mục tiêu
Đọc 235 file CV dạng PDF, trích xuất văn bản thô, và sử dụng Cloud LLM để chuyển đổi văn bản thô thành dữ liệu cấu trúc (JSON) phục vụ cho giao diện Web và các thao tác lọc.

## Chi tiết Công việc
- [x] **Giải nén dữ liệu:** Đã giải nén `archive.zip` thành 115 CV Banking và 120 CV IT vào thư mục `data/cv_resumes/`.
- [x] **Đánh giá chất lượng (Data Quality Analysis):** Đã phân tích 100% là PDF chứa text, độ dài trung bình ~6.391 ký tự. Anonymized cao (rất ít email/phone).
- [ ] **Code script Trích xuất (`scripts/extract_cv_entities.py`):**
  - Đã code xong script sử dụng `pdfplumber` để lấy text.
  - Tích hợp `instructor` và `Pydantic` để ép LLM trả về đúng schema JSON: `name`, `skills`, `experience`, `education`, `certifications`, v.v. Các field thiếu bắt buộc để `null`.
  - Tích hợp **Cơ chế Sandbox**: Bọc text CV trong thẻ `<PAYLOAD>` để chống prompt injection.
- [ ] **Chạy Batch Processing (Cần API Key):**
  - Chạy kịch bản gọi lên Groq API để xử lý toàn bộ 235 file.
  - Lưu kết quả vào `data/cv_resumes/extracted_entities.json`.
  - **Blocker:** Cần người dùng cấu hình `GROQ_API_KEY` vào file `.env`.
