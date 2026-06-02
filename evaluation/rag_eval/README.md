# RAGAS Evaluation (app-backed RAG)

Chạy evaluation bằng RAGAS trên chính RAG stack của project (Qdrant + filter `tenant_id`).

## Chuẩn bị

- Đảm bảo Qdrant đang chạy và đã ingest dữ liệu.
- Thiết lập API key trong `.env` (repo root) hoặc env vars:
  - `GROQ_API_KEY`
  - `GROQ_BASE_URL` (mặc định `https://api.groq.com/openai/v1`)
  - `GROQ_MODEL` (ví dụ `openai/gpt-oss-120b`)
- Dataset JSONL (RAGEval-style) đặt tại:
  - `evaluation/datasets/testset_vi_all_tenants.jsonl`
  - hoặc set `RAGEVAL_JSONL_PATH` trỏ tới file khác.

## Chạy (khuyến nghị dùng conda env `agent`)

```powershell
powershell -ExecutionPolicy Bypass -File .\evaluation\rag_eval\run_evals_agent.ps1
```

Tuỳ chọn:
- `RAG_TOP_K=5`
- `RAG_EVAL_LIMIT=10` (chạy nhanh để ước lượng)

