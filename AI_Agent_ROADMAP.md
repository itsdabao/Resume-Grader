# Roadmap – Chatbot RAG đa kênh (Zalo OA + Messenger) cho Trung tâm Tiếng Anh

Mục tiêu: phát triển sản phẩm chatbot thương mại, hỗ trợ **tư vấn khóa học + chốt lead + handoff tư vấn viên**, vận hành **multi-tenant** (nhiều trung tâm), có thể mở rộng **multi-branch** (nhiều chi nhánh), triển khai trên **Zalo OA và Messenger**.

---

## Nguyên tắc sản phẩm

- **Đúng dữ liệu (no hallucination)**: chỉ trả lời từ tri thức đã ingest; thiếu dữ liệu thì xin phép lấy thông tin để tư vấn viên liên hệ.
- **Lead-first**: mục tiêu tối thiểu lấy được **SĐT**; sau đó hỏi thêm thông tin để tư vấn tốt hơn (progressive profiling).
- **Handoff**: luôn có đường lui sang người thật (CSKH) và ghi nhận trạng thái xử lý.
- **Multi-tenant an toàn**: tách dữ liệu/khóa API/cấu hình theo tenant; kiểm soát truy cập và audit logs.
- **Quan sát được**: đo chất lượng (retrieval/answer), latency, tỉ lệ chốt lead, tỉ lệ handoff.

---

## Phạm vi MVP (bắt buộc để bán được)

- Kênh: **Zalo OA + Messenger** (nhận tin nhắn, trả lời, quản lý session).
- RAG: ingest tài liệu nội bộ trung tâm (PDF/DOCX/MD/TXT/RTF), hybrid search, trích nguồn.
- Guardrails: chống prompt injection cơ bản + lọc nội dung độc hại + scope check (trong/ngoài trung tâm).
- CRM tối thiểu: lưu lead + lịch sử hội thoại + trạng thái xử lý (new/contacted/qualified/closed).
- Admin tối thiểu: upload tài liệu theo tenant/chi nhánh, cấu hình prompt/brand/giờ hoạt động, xem logs.
- Owner Console (local-first): dashboard/logs/handoffs toàn hệ thống + auth hardcode qua env + JWT cookie (`/owner/*`).
- Vận hành: Docker Compose (app + Qdrant + DB), secrets qua env, backup/restore.

### Multi-tenant + Multi-branch (xử lý cả 2 trường hợp)

- **Tenant** = 1 trung tâm (tổ chức). **Branch** = chi nhánh (có thể có hoặc không).
- Nếu trung tâm **chỉ 1 chi nhánh**: mọi kênh và tri thức gắn trực tiếp vào `tenant_id`.
- Nếu trung tâm **nhiều chi nhánh**:
  - Kênh có thể tách theo chi nhánh (mỗi chi nhánh 1 OA/page) hoặc dùng chung một OA/page.
  - Tri thức có thể:
    - **Chung toàn trung tâm** (khóa học, học phí chuẩn, quy định) và
    - **Riêng theo chi nhánh** (địa chỉ, giờ học, lịch khai giảng, ưu đãi theo cơ sở).
  - Thiết kế dữ liệu gợi ý: `tenant_id` + `branch_id` (tùy chọn) trong DB và metadata RAG, để filter đúng ngữ cảnh.

---

## Định hướng kiến trúc SaaS/Agentic (2026)

Phần này có thể áp dụng cho project hiện tại vì hệ thống đã có nền tảng **LlamaIndex + Qdrant + LLM (Gemini)**. Tuy nhiên cần lưu ý các điều kiện:
- `LlamaParse` là dịch vụ/SDK cần cấu hình API key và (thường) cần network để parse PDF phức tạp.
- `BGE-Reranker-v2-m3` chạy local sẽ cần GPU (mục tiêu 3050 như yêu cầu); nên có phương án fallback CPU/không-rerank khi deploy nhỏ.
- `PostgresChatStore`/`SemanticRouterQueryEngine` phụ thuộc phiên bản LlamaIndex; cần pin/upgrade version phù hợp khi triển khai.

### 1) Modular Architecture (mục tiêu refactor)

Chuẩn hoá theo 3 lớp:
- `app/api/`: FastAPI endpoints + adapters kênh (Zalo/Messenger webhooks).
- `app/services/`: RAG pipeline, Agent tools, Router, ingestion service.
- `app/core/`: Config, bootstrap `Settings` (LLM/embeddings), logging, clients (Qdrant/Postgres).

Ghi chú: repo hiện đang có `apps/`, `scripts/`, `src/` → có thể migrate dần sang `app/` theo module boundaries ở trên.

### 2) Modern Ingestion (PDF phức tạp + bảng biểu)

Mục tiêu: ingest PDF “khó” (bảng học phí, lịch học, biểu mẫu) giữ được cấu trúc.
- Dùng `LlamaParse` để parse PDF → Markdown (thay vì chỉ dựa vào reader cơ bản).
- Dùng `MarkdownElementNodeParser` để giữ cấu trúc element (heading/table/list) trước khi chunk.
- Lưu metadata đủ để trace: `tenant_id`, `branch_id`, `doc_id`, `source`, `page`, `element_type`.

### 3) Semantic Router (giảm LLM call + agentic workflow)

Thay router thủ công bằng router ngữ nghĩa của LlamaIndex (ưu tiên `SemanticRouterQueryEngine` nếu phiên bản hỗ trợ; nếu không thì fallback `RouterQueryEngine`).

Nhánh đề xuất:
- `course_info`: vào RAG (retrieval + rerank + citations).
- `general_chat`: gọi LLM trực tiếp (chỉ dùng cho smalltalk/FAQ chung; vẫn có guardrails).
- `enrollment_action`: gọi tools (SQL) để xử lý nghiệp vụ: tính học phí, kiểm tra slot lớp, tạo đăng ký/handoff ticket.

### 3.1) Preprocessing Workflow (Safety & Routing trước RAG2)

Thêm một lớp “gác cổng” chạy trước Router/RAG để giảm rủi ro và giảm số lần gọi LLM:
- **Language Detection**: nhận diện Tiếng Việt / Khác, ưu tiên cực nhanh (mục tiêu ~0.0005s như FastText). Nếu không phải tiếng Việt → trả lời “chưa hỗ trợ” hoặc chuyển sang flow phù hợp.
- **Toxic/Prompt-injection Filter**: dùng model phân loại nhẹ (SVM/SLM) + rule-based để chặn câu hỏi độc hại/prompt injection.
- **Domain Filter (In/Out domain)**: model nhẹ (SVM/SLM) hoặc semantic gate để xác định câu hỏi có nằm trong phạm vi tư vấn; out-domain → handoff xin SĐT thay vì vào RAG/LLM.

### 4) Multi-tenant Qdrant (cô lập dữ liệu)

Yêu cầu bắt buộc: mọi truy vấn vector phải áp `MetadataFilters` theo `tenant_id` (và `branch_id` nếu bật multi-branch), tránh rò dữ liệu chéo khách hàng.

### 5) Advanced Retrieval (độ chính xác ưu tiên)

- Hybrid Search: vector (BGE-M3) + BM25 local, đặt `alpha = 0.5` làm baseline.
- Rerank: tích hợp `BGE-Reranker-v2-m3` (local GPU) để rerank **top 10–20 chunks** rồi chọn **top 3–5** chunks tốt nhất trước khi gửi sang LLM (fallback: tắt rerank hoặc rerank nhẹ trên CPU).

### 6) Conversation Memory (bền vững + tiết kiệm tokens)

- Dùng `PostgresChatStore` để lưu lịch sử hội thoại theo user/session.
- Giới hạn history budget ~1000 tokens (hoặc theo chars) + tóm tắt khi cần để giảm chi phí.

### 7) Evaluation & Admin (đúng tinh thần “thương mại hoá”)

- RAGAS evaluation: `Faithfulness`, `Answer Relevance` (mục tiêu > 0.8).
- Admin dashboard: Avg Time, tỉ lệ hài lòng, tỉ lệ handoff, tỉ lệ capture SĐT, lỗi webhook.

---

## Roadmap tốc chiến đến 16/1 (10 ngày)

Mục tiêu: chuyển từ “xử lý tĩnh” sang “Agentic workflow”, ưu tiên những phần giúp **bán được** và **giảm LLM calls**.

### Giai đoạn 1: Nền tảng SaaS & Ingestion (6/1 - 8/1)

- Day 1: Refactor theo Modular Architecture; bootstrap `Settings`; chuẩn hoá Qdrant + payload index cho `tenant_id` (và `branch_id` nếu bật).
- Day 2: Code ingestion service: `LlamaParse` → Markdown → `MarkdownElementNodeParser` → nodes → Qdrant.
- Day 3: Triển khai Hybrid Search (Vector BGE-M3 + BM25 local), baseline `alpha = 0.5`, log metrics retrieval.

### Giai đoạn 2: Agentic Core & Routing (9/1 - 12/1)

- Day 4: Xây Semantic Router (LlamaIndex router) + lớp Preprocessing Workflow (language detect + toxic/domain filter). Định nghĩa tools:
  - `retrieval_tool` / `course_search_tool` (RAG: thông tin khóa học),
  - `calculator_tool` / `tuition_calculator_tool` (reasoning/tool-first: tính học phí/ưu đãi),
  - `comparison_tool` (so sánh các gói học/khóa học theo tiêu chí),
  - `enrollment_create_ticket_tool` (SQL insert ticket/handoff).
- Day 5: Xử lý câu hỏi suy luận nghiệp vụ (ví dụ tổng học phí sau giảm giá) bằng tool + kiểm tra input; không “thả” LLM tự tính, và không cần xuất chain-of-thought cho người dùng.
- Day 6–7: Memory bền vững với `PostgresChatStore`; giới hạn 1000 tokens; tóm tắt theo session khi quá ngưỡng.

### Giai đoạn 3: Evaluation & UI Polish (13/1 - 16/1)

- Day 8: Tích hợp RAGAS evaluation; set target `Faithfulness` & `Answer Relevance` > 0.8 cho bộ test chuẩn.
- Day 9: Hoàn thiện Dashboard Admin (theo mẫu luận văn): Avg Time, tỉ lệ hài lòng, tỉ lệ handoff, logs theo tenant.
- Day 10 (16/1): Đóng gói Docker; kiểm tra WebSocket; smoke test end-to-end; chuẩn bị RELEASE notes.
---

## Timeline gợi ý (12 tuần)

> Có thể rút còn 6–8 tuần nếu chốt scope nhỏ và chỉ 1 tenant pilot.

### Giai đoạn 0 (Tuần 1): Chốt yêu cầu & thiết kế

**Deliverables**
- PRD v1: use-cases, intent chính, trường lead, quy trình handoff, KPI.
- Thiết kế kiến trúc: Core RAG + Channel Adapters + Admin + DB.
- Chuẩn hóa dữ liệu mẫu của 1 trung tâm (tài liệu học phí/lịch/khóa học/quy định).

**Checklist**
- Danh sách intent: học phí, lịch học, ưu đãi, lộ trình, test đầu vào, địa chỉ, giờ làm việc.
- Kịch bản hỏi ngược làm rõ + kịch bản chốt lead.

### Giai đoạn 1 (Tuần 2–3): “Core Platform” (chuẩn hóa để mở rộng)

**Mục tiêu**: tách lõi khỏi kênh, làm nền multi-tenant & lưu trữ.

**Công việc**
- Thiết kế schema DB (Postgres): tenant, branch, channel_account, conversation, message, lead, document_job.
- Chuẩn hóa API nội bộ: `POST /v1/chat` (core), trả về `{answer, sources, lead_update?, handoff?}`.
- Chuẩn hóa cấu hình tenant/branch: system prompt, brand voice, giờ hoạt động, chính sách.
- Routing theo `branch_id` (nếu có): lấy từ kênh (OA/page) hoặc hỏi người dùng chọn cơ sở.
- Logging chuẩn + correlation id.

**Done when**
- Có thể chạy local 1 tenant: ingest → chat core → ghi message/lead vào DB.

### Giai đoạn 2 (Tuần 4–5): Ingestion & quản trị tri thức

**Mục tiêu**: ingest bền vững cho tài liệu trung tâm.

**Công việc**
- Pipeline ingest theo tenant/branch: versioning, re-ingest, xóa/rollback.
- Metadata bắt buộc: `tenant_id`, `branch_id` (optional), `source`, `doc_id`, `chunk_id`, `updated_at`.
- UI/endpoint admin tối thiểu: upload file, xem trạng thái job, xem tài liệu đã ingest.
- Bộ kiểm tra chất lượng ingest: số chunks, top terms, phát hiện file lỗi.

**Nâng cấp đề xuất**
- OCR PDF scan (nếu cần) + chuẩn hóa markdown cho bảng học phí.

### Giai đoạn 3 (Tuần 6–8): Kênh Zalo OA + Messenger (Adapters)

**Mục tiêu**: đưa chatbot ra kênh thật, đảm bảo idempotency, rate limit, bảo mật webhook.

**Zalo OA Adapter**
- Webhook nhận tin nhắn → map vào core chat.
- Xác thực chữ ký (nếu nền tảng yêu cầu), chống replay.
- Quản lý session theo `user_id` + `tenant` (+ `branch` nếu có).

**Messenger Adapter**
- Webhook nhận message/postback → map vào core chat.
- Xử lý quick replies / buttons cho flow chốt lead.

**Tính năng kênh**
- Template trả lời: ngắn gọn + gạch đầu dòng + trích nguồn.
- Nút “Gặp tư vấn viên” → tạo ticket/handoff.
- Hạn chế spam: rate limit theo user/tenant.

### Giai đoạn 4 (Tuần 9–10): Guardrails & chất lượng

**Mục tiêu**: an toàn khi bán cho doanh nghiệp.

**Công việc**
- Prompt-injection / toxic filter (rule-based + model nhẹ tùy điều kiện).
- Scope classifier: câu hỏi ngoài trung tâm → xin phép lưu SĐT/handoff.
- Policy: không trả lời ngoài phạm vi; không bịa.
- “Answer verifier” cơ bản: nếu không có nguồn phù hợp → trả lời thiếu dữ liệu/handoff.

**Đánh giá (theo tinh thần tài liệu mẫu)**
- Retrieval: Hit@K, MRR, NDCG (tập câu hỏi chuẩn).
- Latency: p50/p95 theo kênh.
- Business KPI: lead capture rate, qualified rate, handoff rate.

### Giai đoạn 5 (Tuần 11–12): Productionization

**Công việc**
- Auth: API key per tenant + RBAC admin.
- Observability: dashboard logs/metrics, cảnh báo lỗi webhook.
- Backup/restore: Qdrant + DB.
- Hardening: CORS allowlist, secret management, input validation.
- Tài liệu triển khai: Docker Compose + hướng dẫn cấu hình kênh.
- Chế độ triển khai **bật/tắt on-premise**: cloud mặc định; on-prem Docker Compose, chọn LLM theo ràng buộc dữ liệu.

**Go-live checklist**
- 1–3 tenant pilot chạy thật 1–2 tuần.
- Quy trình CSKH: ai nhận handoff, SLA, kịch bản xử lý.

---

## Backlog theo Epic (để chia việc)

### Epic A – Product & Conversation
- Intent catalog + flow hỏi ngược.
- Lead flow (từng trường 1) + định dạng `LEAD_DATA`.
- FAQ fallback + handoff.

### Epic B – Core RAG
- Chuẩn hóa chunking cho bảng học phí/lịch học.
- Hybrid tuning (alpha/top_k), rerank, prompt budget.
- Citation chuẩn (nguồn + đoạn).

### Epic C – Multi-tenant / Multi-branch
- Tenant isolation: filter bắt buộc khi query.
- Branch routing + dữ liệu chung/riêng theo chi nhánh.
- Quota theo tenant, logging per tenant.

### Epic D – Admin
- Upload/ingest/re-ingest.
- Quản lý prompt & cấu hình tenant/branch.
- Dashboard: số hội thoại, lead, lỗi.

### Epic E – Channel Adapters
- Zalo OA: webhook, reply, quick actions.
- Messenger: webhook, buttons, quick replies.
- Handoff routing.

### Epic F – Security & Compliance
- Auth/RBAC, rate limit, audit log.
- PII encryption/retention policy.

---

## Scope đã chốt (theo trả lời của bạn)

1. **Multi-branch**: xử lý được cả 1 chi nhánh và nhiều chi nhánh (`tenant_id` + `branch_id`).
2. **Lead tối thiểu: SĐT**; các trường khác là tùy chọn để tư vấn tốt hơn (progressive profiling).
3. **Handoff/CRM**:
   - Mức tối thiểu: đánh dấu `handoff=true`, lưu lead + lịch sử, thông báo cho tư vấn viên trên kênh.
   - Mức tiêu chuẩn để bán: tạo ticket có trạng thái, người phụ trách, SLA.
   - Mức nâng cao: đồng bộ CRM (Google Sheet/HubSpot/Zoho/…) hoặc webhook nội bộ.
4. **On-premise bật/tắt**: tùy khách hàng; thiết kế deployment mode để chuyển đổi linh hoạt.
5. **LLM linh hoạt, ưu tiên độ chính xác**: hỗ trợ nhiều provider (Gemini/OpenAI/local) và bắt buộc có lớp đo lường + guardrails (“không có nguồn thì không trả lời”).
