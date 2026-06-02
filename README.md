# Multi-Domain AI RAG System for CV Parsing & Candidate Retrieval

An Agentic Retrieval-Augmented Generation (RAG) system built to handle unstructured resumes (CVs) across multiple domains (Information Technology & Banking). The system extracts structured entities from both digitally generated and scanned PDFs, stores them in a Vector Database, and provides a chat interface for HR professionals to discover the right candidates intuitively.

---

## 🚀 Key Features

*   **One RAG, Multiple Domains:** A unified Qdrant vector space seamlessly handles both IT and Banking candidates without manual filtering.
*   **Ultra-low Latency Agentic Search:** Using regex-based fast routing and parameter extraction coupled with **Groq API** (Llama 4) for extremely fast candidate synthesis (~28s total latency).
*   **Advanced OCR Fallback:** Automatically detects scanned (image-based) PDFs and triggers **Google Gemini 3.5 Flash Vision API** to accurately extract text while maintaining layout integrity.
*   **Modern Web UI:** A sleek, deployable React interface with dedicated panels for Chatting and Document Management.

---

## 🛠️ Technology Stack

### Backend (Extraction & RAG Engine)
*   **Framework:** FastAPI (Python)
*   **Vector Database:** Qdrant (Local instance for privacy & low latency)
*   **Embeddings:** `intfloat/multilingual-e5-large` (Dense vector representations)
*   **LLM Orchestration:** `llama-index`
*   **Core LLMs:** 
    *   **Agent/Routing:** Groq API (OpenAI-compatible) or Google Gemini for candidate synthesis.
    *   **Vision OCR:** Google Gemini Vision (for scanned PDFs).
*   **Document Parsers:** `PyMuPDF` (fitz) & `pdfplumber`.

### Frontend
*   **Framework:** React / Vite (or Next.js)
*   **Styling:** TailwindCSS
*   **Features:** Real-time chat streaming, document upload interface.

---

## 🏗️ System Workflow (End-to-End)

1.  **Ingestion & Parsing:**
    *   HR uploads a CV (PDF/DOCX). 
    *   The system attempts standard text extraction (`pdfplumber`).
    *   *Edge Case Handling:* If the document yields less than 100 characters (e.g., Scanned Image), the system rasterizes the PDF and routes it to the **Gemini Vision API** for visual text extraction.
2.  **Entity Extraction:**
    *   The raw text is fed into an LLM with a strict JSON schema prompt to extract structured data: `Name`, `Skills`, `Experience Level`, `Domain` (IT/Banking), etc.
3.  **Vector Storage:**
    *   The extracted JSON payload is embedded into semantic vectors and upserted into Qdrant.
4.  **Retrieval & Chat Interaction:**
    *   User inputs a query: *"Find me 3 Senior AI Engineers with Python skills."*
    *   The **Regex-based Router** identifies the intent and extracts search parameters (e.g., Python, 3+ years experience).
    *   Qdrant performs a similarity search (`query_points`), returning the most relevant CVs.
    *   The Groq LLM synthesizes a natural, concise answer summarizing why these candidates match.

---

## 📊 Evaluation & Performance Metrics

To ensure production readiness, the system includes an evaluation module (`scripts/evaluate_rag.py`) focusing on:
*   **Context Precision:** Ensuring retrieved CVs strictly match the query constraints (e.g., 3+ years experience).
*   **Context Recall:** Avoiding missed candidates by relying on dense semantic vectors rather than brittle keyword matching.
*   **Latency:** Utilizing local models for embeddings to reduce external API roundtrips.

---

## 💻 Local Setup & Deployment

### Prerequisites
*   Python 3.11+ (Conda recommended)
*   Node.js 18+

### 1. Environment Setup
Copy the `.env.example` file to `.env` and configure your API keys:
```bash
cp .env.example .env
```
Fill in the necessary API keys in `.env`:
- `GROQ_API_KEY` (For Chat & Synthesis)
- `GOOGLE_API_KEY` (For Scanned CV OCR Fallback via Gemini)

### 2. Start the Backend
```bash
conda activate agent
pip install -r requirements.txt
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```
Visit `http://localhost:8000/docs` to verify the API Swagger documentation.

### 3. Start the Frontend
```bash
cd web
npm install
npm run dev
```
Navigate to the local dev URL (usually `http://localhost:5173` or `http://localhost:3000`) to interact with the RAG System.
