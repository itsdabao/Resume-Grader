# Phase 4: Evaluation (Đánh giá)

## Mục tiêu
Kiểm chứng chất lượng của hệ thống RAG bằng cách sử dụng bộ khung RAGAS có sẵn của repo.

## Chi tiết Công việc
- [ ] **Tạo dữ liệu Test (`evaluation/datasets/testset_cv.jsonl`):**
  - Viết 20 câu hỏi truy vấn chuyên môn (10 cho Banking, 10 cho IT).
  - Khai báo rõ Ground Truth (Keypoints) để LLM Giám khảo (Judge LLM) có thể chấm điểm.
- [ ] **Chạy đánh giá (`evaluation/rag_eval/evals.py`):**
  - Cập nhật script cũ để trỏ tới `testset_cv.jsonl`.
  - Cập nhật prompt chấm điểm cho phù hợp với ngữ cảnh nhân sự.
  - Đo lường và xuất file báo cáo độ chính xác (`.csv`).
