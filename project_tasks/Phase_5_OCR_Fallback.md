# Phase 5: OCR Fallback (Dự phòng cho Ảnh Scan)

## Mục tiêu
Giải quyết tình trạng Out-of-memory (OOM) nếu bắt buộc phải đọc các file scan hoặc ảnh CV, đồng thời đạt 10% điểm thưởng của bài test. Phương án sử dụng toàn bộ CPU.

## Chi tiết Công việc
- [ ] **Tích hợp `pytesseract` & `OpenCV`:**
  - Viết logic xử lý chuyển đổi PDF sang ảnh (dùng `pdf2image` hoặc `PyMuPDF`).
  - Dùng OpenCV để chuyển ảnh sang ảnh xám (grayscale) và nhị phân hóa (thresholding) tăng độ nét.
  - Gọi Tesseract OCR để rút trích chữ ra khỏi ảnh.
- [ ] **Cơ chế Kích hoạt (Trigger):**
  - Cập nhật vào hàm `get_pdf_text()` trong `extract_cv_entities.py`.
  - Chỉ kích hoạt hàm xử lý ảnh NẾU độ dài văn bản trích xuất bằng pdfplumber trả về < 50 ký tự.
- [ ] **Test với file rác:**
  - Bỏ 1 ảnh JPG CV mẫu vào để đảm bảo hệ thống fallback hoạt động đúng.
