# Phase 2: RAG Pipeline Optimization (Tối ưu Pipeline RAG)

## Mục tiêu
Đưa dữ liệu văn bản CV vào hệ thống Vector Store, tinh chỉnh Agentic Router để hiểu được ngữ cảnh tìm kiếm CV của ngành nhân sự.

## Chi tiết Công việc
- [x] **Cập nhật Cấu hình (`app/core/config.py`):** 
  - Đã thêm cờ `CV_MODE=True`.
  - Thiết lập Hybrid provider: `HEAVY_LLM_PROVIDER=groq`, `ROUTER_LLM_PROVIDER=llama_cpp`.
  - Cho phép truy vấn xuyên danh mục bằng cách cập nhật `REQUIRE_TENANT_ID`.
- [x] **Tối ưu Semantic Router (`app/services/agentic/router.py`):**
  - Đã code lại logic router.
  - Gọi Local SLM (file `qwen2.5-3b-instruct-q4_k_m.gguf` trong `d:\IMT_test\models`) qua `llama_cpp`.
  - Các intent mới: `cv_search`, `skills_analytics`, `general_chat`.
  - Trích xuất metadata `tenant_id` là `banking` hoặc `it` ngay từ câu hỏi.
- [ ] **Viết script Ingestion (`scripts/ingest_cv.py`):**
  - Viết script bao bọc lại `run_ingestion()` của hệ thống cũ.
  - Đọc 115 file Banking gắn `tenant_id=banking`.
  - Đọc 120 file IT gắn `tenant_id=it`.
- [ ] **Thực thi Ingestion:** Chạy script để nạp dữ liệu qua BGE-M3 vào cơ sở dữ liệu Qdrant.
