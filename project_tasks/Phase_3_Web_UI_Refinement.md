# Phase 3: Web UI Refinement (Tinh chỉnh Giao diện Web)

## Mục tiêu
Thay đổi giao diện của hệ thống cũ (vốn dùng cho trung tâm tiếng Anh) thành giao diện hỗ trợ chuyên dụng cho việc tuyển dụng, quét CV. Vẫn sử dụng kiến trúc FastAPI và HTML/JS thuần.

## Chi tiết Công việc
- [ ] **Giao diện Chat (`web/agent.html` & `web/agent.js`):**
  - Thay đổi logo, tiêu đề thành "Trợ lý ảo Tuyển dụng (CV RAG System)".
  - Thêm một dropdown filter để lọc "Tất cả ngành", "Banking", "IT".
  - Sửa các câu hỏi gợi ý (Suggest Queries) thành các câu phù hợp: "Tìm ứng viên IT giỏi Python", "Thống kê kỹ năng ngành Banking", v.v.
- [ ] **Giao diện Admin (`web/admin.html`):**
  - Chuyển đổi công năng thành "Document Manager" (Quản lý Hồ sơ CV).
  - Viết code để đọc file `extracted_entities.json` và hiển thị danh sách CV ra bảng.
  - (Bonus) Hiển thị biểu đồ phân bổ kỹ năng.
- [ ] **API Endpoint (`app/api/main.py`):**
  - Cập nhật thêm API để web có thể lấy được dữ liệu thống kê từ JSON nếu cần.
