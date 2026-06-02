# Viet RAG Multitenant Assistant

Trợ lý hỏi‑đáp tiếng Việt dựa trên RAG 2.0, dùng LlamaIndex + Qdrant + Hybrid Retrieval (Vector + BM25) + Agentic Tools + Postgres Memory. Hệ thống hỗ trợ multi‑tenant (nhiều khách hàng), hybrid search, rerank nhẹ bằng cosine, có CLI + backend FastAPI + web demo + admin dashboard.

## Tính năng chính

- Ingest tài liệu (PDF/DOCX/TXT/MD/RTF) → chunk → embedding → Qdrant.
- Multi‑tenant isolation bằng metadata filters (`tenant_id`, `branch_id`) để tránh “râu ông nọ chắp cằm bà kia”.
- Hybrid retrieval: kết hợp vector search (Qdrant) + lexical search (BM25 local).
- Rerank nhẹ bằng cosine để chọn top contexts tốt nhất trước khi đưa vào prompt.
- Agentic workflow: preprocessing → routing → tool execution (tính học phí/tổng thanh toán, so sánh, tạo ticket/handoff, RAG course_search).
- Persistent memory (Postgres): rolling summary + recent buffer + entity_memory theo session.
- Admin Dashboard: Avg time, p95, satisfaction rate, handoff rate, logs theo tenant.

---

## Công nghệ / kỹ thuật / model / thuật toán đang dùng (và vai trò)

### 1) FastAPI + WebSocket (Backend API)
- **FastAPI** (`app/api/main.py`): cung cấp HTTP endpoint `/query` và các endpoint admin (`/admin/api/*`).
- **WebSocket** (`/ws/query`): stream câu trả lời theo chunk để web demo có trải nghiệm giống ChatGPT.
- **CORS middleware**: cho phép web demo gọi API trong môi trường dev.
- **Web apps (web tĩnh)**: Agent Chat (`/agent`) và Owner Console (`/owner`) được serve bởi backend.

### 2) LlamaIndex (RAG framework / orchestration)
- **VectorStoreIndex + Settings**: chuẩn hoá cách load vector store, embeddings, LLM, và gọi pipeline RAG.
- **Query pipeline**: dựng prompt từ system prompt + lịch sử + contexts + few-shot (in-context RALM).
- **Abstraction LLM**: chạy với nhiều provider (Gemini/Groq/OpenAI‑compatible) tùy env vars.

### 3) Qdrant (Vector Database)
- **Qdrant** (`localhost:6333`): lưu vector embeddings + payload metadata.
- **Payload filter**: bắt buộc lọc theo `tenant_id` (và `branch_id` nếu bật) để đảm bảo cô lập dữ liệu SaaS.

### 4) Embedding Model: `BAAI/bge-m3`
- Vai trò: biến text thành vector 1024‑d (cosine) để truy vấn ngữ nghĩa trên Qdrant.
- Dùng cho: retrieval (vector search), rerank cosine (query vs context) và chọn few-shot theo similarity.

### 5) BM25 (Lexical Retrieval)
- Vai trò: truy hồi theo từ khóa (lexical) để “bám chữ”, hữu ích với câu hỏi chứa mã/thuật ngữ/định danh.
- Dùng cho: hybrid search (kết hợp với vector bằng `HYBRID_ALPHA`) và fallback khi không có index.

### 6) Hybrid Retrieval (Vector + BM25) + Fusion
- **Hybrid alpha**: cân bằng độ “ngữ nghĩa” (vector) và độ “bám chữ” (BM25).
- Vai trò: tăng recall trong tài liệu thực tế (đặc biệt là bảng học phí/lịch/định dạng không chuẩn).

### 7) Rerank nhẹ bằng cosine
- Vai trò: sau khi retrieve top‑K, tính cosine(query_embedding, context_embedding) để chọn lại top contexts tốt hơn.
- Lợi ích: giảm nhiễu, giảm token, tăng faithfulness cho câu trả lời.

### 8) In‑Context RALM (Few‑shot + Context + History)
- Vai trò: nhúng few‑shot examples phù hợp + contexts + (optional) history để “dạy” LLM trả lời đúng format, đúng scope.
- Lợi ích: ổn định văn phong, giảm prompt engineering thủ công rời rạc.

### 9) Agentic Routing + Tool‑first 
- **Preprocessing**: language detect, toxic filter, phone extraction.
- **Router**: quyết định route (smalltalk/out_of_domain/course_search/tuition_calculator/comparison/create_ticket) để giảm LLM call và tăng tính “business”.
- **Tool-first**:
  - `tuition_calculator_tool`: tính **tổng thanh toán** (học phí + phí phụ), xử lý scope giảm giá (“giảm gói học” vs “giảm tổng”).
  - `comparison_tool`: so sánh nhanh 2 gói/khóa theo evidence.
  - `create_ticket_tool`: handoff sang tư vấn viên + lưu ticket.
  - `course_search_tool`: RAG trả lời đầy đủ (LLM‑backed).

### 10) LLM‑backed Finance Extractor (chống nhiễu tiền rác)
- `extract_financials_with_llm` + `refine_extracted_fees`: phân loại **học phí chính** vs **phí phụ** ngay cả khi OCR rớt “VND/đ”.
- Vai trò: tăng độ đúng của công cụ tính tiền khi trong tài liệu có nhiều con số (ví dụ 9.000.000 vs 300.000).

### 11) Postgres + SQLAlchemy (Memory bền vững + Analytics)
- **Conversation memory (Day 6–7)**:
  - `chat_sessions`: `entity_memory` (JSONB), `rolling_summary`, `recent_messages_buffer`.
  - Cơ chế: giới hạn budget (~1000 tokens), tự roll‑up summary, giữ last N turns.
- **Analytics (Day 9)**:
  - `request_traces`: 1 dòng / request (latency, route, sources_count, tool_metadata…).
  - `user_feedback`: thumbs up/down để tính satisfaction.
  - `handoff_tickets`: ticket/handoff để tính handoff rate.

### 12) Web UI (HTML/CSS/JS thuần)
- `web/frontend_test.html` + `web/appjs.js`: landing + live demo streaming (WebSocket), có session_id (memory), feedback (👍/👎), hiển thị trace/route.
- `web/admin.html` + `web/admin.js`: dashboard KPI + logs + handoffs theo tenant.
- `web/agent.html` + `web/agent.js`: Agent Chat UI (tối giản, streaming).
- `web/owner.html` + `web/owner.js`: Owner Console UI (dashboard/logs/handoffs, login bằng env + JWT cookie).

### 13) Evaluation 
- Thư mục `evaluation/`: gom các script/artefact evaluation (RAGAS) để dễ quản lý và tránh push nhầm dữ liệu.

---

## 1. Chuẩn bị môi trường

Yêu cầu:
- Python 3.11+ (khuyến nghị chạy trong conda env `agent` như repo đang dùng)
- Docker (để chạy Qdrant)
- Postgres (để bật memory + dashboard; nếu không có Postgres thì chỉ chạy được phần core RAG, không có dashboard)

Khuyến nghị tạo môi trường ảo riêng (ví dụ conda):

Tạo file `.env` ở root:

```bash
# Postgres 
DATABASE_URL=postgresql+psycopg2://admin:123@localhost:5432/agent_memory
# Hoặc dùng SQLite cho local dev (nhanh, không cần Postgres):
# DATABASE_URL=sqlite:///./data/agent.db

# (Optional) ép provider để tránh tự fallback sang Gemini/OpenAI
LLM_PROVIDER=groq

# Primary (recommended): Groq (OpenAI-compatible)
GROQ_API_KEY=your_groq_api_key_here  # không để dấu nháy, không cắt ngắn
GROQ_MODEL=openai/gpt-oss-120b
GROQ_BASE_URL=https://api.groq.com/openai/v1
# Groq được gọi qua OpenAI-compatible endpoint `GROQ_BASE_URL` (cách 2).

# Local/Remote OpenAI-compatible (chạy local model)
# Ví dụ base_url:
# - LM Studio: http://localhost:1234/v1
# - vLLM:      http://localhost:8001/v1
# - Ollama:    http://localhost:11434/v1 (nếu bật OpenAI-compat)
# Dùng:
# LLM_PROVIDER=openai_compat
# OPENAI_COMPAT_BASE_URL=http://localhost:1234/v1
# OPENAI_COMPAT_MODEL=your-local-model-id
# OPENAI_COMPAT_API_KEY=local

# llama-cpp-python (in-process GGUF) - load GGUF trực tiếp trong backend
# LLM_PROVIDER=llama_cpp
# LLAMA_CPP_MODEL_PATH=models\\qwen2.5-3b-instruct-q4_k_m.gguf
# LLAMA_CPP_N_GPU_LAYERS=-1
# LLAMA_CPP_N_CTX=2048
# Optional:
# LLAMA_CPP_CHAT_FORMAT=chatml
# LLAMA_CPP_N_THREADS=0
# LLAMA_CPP_TEMPERATURE=0.2
# LLAMA_CPP_MAX_TOKENS=1024
# LLAMA_CPP_VERBOSE=0

# Optional: LlamaParse (Modern Ingestion cho PDF phức tạp)
# LLAMA_CLOUD_API_KEY=your_llama_cloud_key_here

# Optional fallback: Gemini
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite

# --- Owner Console (local-first) ---
OWNER_USERNAME=owner
OWNER_PASSWORD=owner_password_here
JWT_SECRET=change_me_to_a_long_random_string
# Optional (minutes), default 1440 (1 day)
JWT_EXPIRE_MIN=1440

# Optional: chỉ bật Owner UI, bỏ qua init RAG/LLM khi startup
# RAG_INIT_ON_STARTUP=0
```

## 2. Chạy Qdrant

Chạy Qdrant bằng Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Mặc định code trỏ tới `localhost:6333` (có thể chỉnh trong `app/core/config.py`).

## 3. Ingest dữ liệu

Ingest toàn bộ file trong `data/knowledge_base`:

```bash
python scripts/ingest.py --auto-from-filenames
```

- Tên file dạng `tenant_*.pdf` sẽ được suy ra `tenant_id`.
- Hoặc ingest thủ công:

```bash
python scripts/ingest.py --tenant brightpathacademy --file data/knowledge_base/tenant_brightpathacademy.pdf
```

Nếu PDF có bảng biểu/phức tạp và bạn có `LLAMA_CLOUD_API_KEY`, có thể bật LlamaParse:

```bash
python scripts/ingest.py --pdf-engine llamaparse --tenant brightpathacademy --file data/knowledge_base/tenant_brightpathacademy.pdf
```

## 4. Chạy CLI hỏi‑đáp

Sau khi ingest và Qdrant đã chạy:

```bash
python scripts/query.py --mode hybrid_rerank --tenant brightpathacademy
```

- Gõ câu hỏi, `exit` để thoát, `/reset` để xoá lịch sử hội thoại.
- Các mode:
  - `vector` – chỉ dùng vector search.
  - `hybrid` – vector + BM25, không rerank cosine.
  - `hybrid_rerank` – hybrid + rerank cosine (đề xuất).

## 5. Chạy backend FastAPI

Khởi động backend:

```bash
uvicorn app.api.main:app --reload --port 8000
```

Truy cập:
- Docs tự động: `http://localhost:8000/docs`
- Agent Chat (WebSocket streaming): `http://localhost:8000/agent`
- Owner Console (login + dashboard/logs/handoffs): `http://localhost:8000/owner`
- Admin Dashboard (Day 9): `http://localhost:8000/admin`
- Semantic Router (trả JSON tool call hoặc text): `POST http://localhost:8000/semantic`
- Web demo (landing + streaming chat): `http://localhost:8000/static/frontend_test.html`
- Root mặc định sẽ redirect về Agent Chat: `http://localhost:8000/`
- Test nhanh endpoint `/query` với body mẫu:

```json
{
  "question": "Trung tâm BrightPath có những chương trình nào?",
  "tenant_id": "brightpathacademy",
  "branch_id": null,
  "session_id": "brightpathacademy:web:demo01",
  "history": []
}
```

Backend sẽ trả:

```json
{
  "answer": "...",
  "sources": ["tenant_brightpathacademy.pdf", "..."],
  "trace_id": "....",
  "time_ms": 123.4,
  "route": "course_search"
}
```

### Ghi chú về LLM API key
- Nếu `GROQ_API_KEY`/`GOOGLE_API_KEY` sai hoặc thiếu, hệ thống vẫn có thể retrieve được context nhưng không gọi được LLM (thường báo 401/invalid_api_key).
- Muốn chat trả lời đầy đủ: cần API key hợp lệ trong `.env` và restart `uvicorn`.

## 6. Cấu trúc thư mục

- `app/api/main.py` – FastAPI backend (HTTP `/query`, WS `/ws/query`, admin `/admin` + `/admin/api/*`).
- `app/core/` – config, bootstrap Settings (LLM/embeddings), adapters provider.
- `app/services/rag_service.py` – entrypoint RAG dùng chung (CLI/backend), tích hợp memory khi có `tenant_id` + `session_id`.
- `app/services/rag/` – hybrid retrieval + rerank + in-context RALM (few-shot + contexts + history).
- `app/services/agentic/` – preprocessing, routing, tools (tuition/comparison/course_search/create_ticket), fee extractor.
- `app/services/memory/` – Postgres memory store + rolling summary + entity memory.
- `app/services/analytics/` – lưu trace/feedback/handoff để dashboard query KPI.
- `scripts/` – ingest/query/eval utilities (Day 8 eval, tool runner…).
- `web/` – landing + live demo + admin dashboard (HTML/CSS/JS thuần).
- `data/` – knowledge_base, cache nodes, eval artifacts, docs tham khảo.
- `src/` – (nếu có) phần code cũ/prototype; core hiện tại ưu tiên `app/`.

## 7. Ghi chú phát triển

Các ý tưởng, roadmap và log phát triển chi tiết nằm trong:
- `data/giai đoạn phát triển.docx`
- `data/Log_phat_trien.docx`
- `ROADMAP.md` (Day 1 → Day 10)

Đây là nơi mô tả các giai đoạn RAG 1.0 → 2.0, hybrid, multi-tenant và kế hoạch đánh giá. 

---

## 8. Evaluate bằng RAGAS (tùy chọn)

### 8.1 Chuẩn bị
- Chạy Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
- Ingest dữ liệu trước khi eval (ví dụ): `python scripts/ingest.py --auto-from-filenames`
- (Khuyến nghị) Có `DATABASE_URL` nếu dùng `--use-memory` (memory/Postgres).
- Chạy script RAGAS trong thư mục `evaluation/` (khuyến nghị dùng conda env `agent`).

RAGAS cần evaluator LLM. Repo cấu hình ưu tiên Groq qua env:
- `GROQ_API_KEY`, `GROQ_BASE_URL`, `GROQ_MODEL` (ví dụ `openai/gpt-oss-120b`)

### 8.2 Input schema
File JSONL, mỗi dòng là 1 dict. Tối thiểu:
- `tenant_id` (hoặc truyền `--tenant`)
- `question`

Các field tùy chọn:
- `id`, `branch_id`, `session_id`
- `ground_truth` (string hoặc list) để tính các metric liên quan “correctness/recall”
- `expected_sources`, `expect_route_one_of`, `must_include` để chạy các “cheap checks”

Ví dụ: `data/eval/ragas_cases.jsonl.example`

### 8.3 Chạy eval
Xem hướng dẫn và chạy script tại `evaluation/README.md`.
