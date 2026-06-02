# Evaluation

Thư mục này gom tất cả thành phần liên quan đến **evaluation** để dễ quản lý và tránh push nhầm dữ liệu khách hàng lên GitHub.

## Cấu trúc

- `evaluation/rag_eval/`: pipeline **RAGAS** (LLM-as-judge) chạy trực tiếp trên RAG thật của project (Qdrant + filter `tenant_id`).
- `evaluation/scripts/`: script tiện ích (tạo testset JSONL, export markdown → JSONL, v.v.).
- `evaluation/latex/`: các bảng/đoạn LaTeX để import vào Overleaf (phần 3.6).
- `evaluation/datasets/`: nơi đặt các file JSONL/CSV phục vụ evaluation (mặc định bị ignore khỏi git).

## Chạy RAGAS evaluation (khuyến nghị dùng conda env `agent`)

```powershell
conda run -n agent python evaluation/rag_eval/evals.py
```

Biến môi trường hay dùng:
- `GROQ_API_KEY`, `GROQ_BASE_URL`, `GROQ_MODEL` (ví dụ `openai/gpt-oss-120b`)
- `RAGEVAL_JSONL_PATH` (đường dẫn tới JSONL testset)
- `RAG_TOP_K` (mặc định `5`)
- `RAG_EVAL_LIMIT` (0 = chạy full)

## Overleaf

Upload `evaluation/latex/3_6_evaluation_tables.tex` và import bằng:

```tex
\input{evaluation/latex/3_6_evaluation_tables}
```
