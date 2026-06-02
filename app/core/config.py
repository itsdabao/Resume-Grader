from __future__ import annotations

import os
from pathlib import Path

from qdrant_client.http.models import Distance


PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    # Load repo-root `.env` so env-driven settings work even when `cwd` is different.
    load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env"))
except Exception:
    pass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# Runtime environment
APP_ENV = (os.getenv("APP_ENV") or os.getenv("ENV") or "dev").strip().lower()
IS_PRODUCTION = APP_ENV in ("prod", "production")

# --- CV Resume RAG Mode (Hardware Optimized) ---
CV_MODE = _env_bool("CV_MODE", True)
CV_DATA_PATH = str(PROJECT_ROOT / "data" / "cv_resumes")
CV_ENTITIES_PATH = str(PROJECT_ROOT / "data" / "cv_resumes" / "extracted_entities.json")

# Cloud API for Heavy Lifting (Extraction / Generation)
HEAVY_LLM_PROVIDER = (os.getenv("HEAVY_LLM_PROVIDER") or "groq").strip()

# Local SLM for Routing
ROUTER_LLM_PROVIDER = (os.getenv("ROUTER_LLM_PROVIDER") or "llama_cpp").strip()
LLAMA_CPP_MODEL_PATH = str(os.getenv("LLAMA_CPP_MODEL_PATH") or "d:\\IMT_test\\models\\qwen2.5-3b-instruct-q4_k_m.gguf")
LLAMA_CPP_N_CTX = int(os.getenv("LLAMA_CPP_N_CTX") or "2048")
VRAM_LIMIT_MODE = _env_bool("VRAM_LIMIT_MODE", True)
# -----------------------------------------------

# HTTP/CORS security controls
_cors_raw = (os.getenv("CORS_ALLOW_ORIGINS") or "").strip()
if _cors_raw:
    CORS_ALLOW_ORIGINS = [x.strip() for x in _cors_raw.split(",") if x.strip()]
else:
    CORS_ALLOW_ORIGINS = (
        []
        if IS_PRODUCTION
        else [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )
CORS_ALLOW_CREDENTIALS = _env_bool("CORS_ALLOW_CREDENTIALS", True)

_cors_methods_raw = (os.getenv("CORS_ALLOW_METHODS") or "").strip()
CORS_ALLOW_METHODS = (
    [x.strip().upper() for x in _cors_methods_raw.split(",") if x.strip()]
    if _cors_methods_raw
    else ["GET", "POST", "OPTIONS"]
)

_cors_headers_raw = (os.getenv("CORS_ALLOW_HEADERS") or "").strip()
CORS_ALLOW_HEADERS = (
    [x.strip() for x in _cors_headers_raw.split(",") if x.strip()]
    if _cors_headers_raw
    else ["Authorization", "Content-Type", "X-API-Key", "X-Requested-With"]
)

if IS_PRODUCTION and CORS_ALLOW_ORIGINS == ["*"]:
    import warnings
    warnings.warn(
        "CORS_ALLOW_ORIGINS='*' in production is a security risk! "
        "Set CORS_ALLOW_ORIGINS to your actual domain(s).",
        stacklevel=1,
    )

# Access control policies
ADMIN_REQUIRE_OWNER_AUTH = _env_bool("ADMIN_REQUIRE_OWNER_AUTH", True)
ENABLE_PUBLIC_QUERY_ENDPOINT = _env_bool("ENABLE_PUBLIC_QUERY_ENDPOINT", False)
PUBLIC_QUERY_API_KEY = (os.getenv("PUBLIC_QUERY_API_KEY") or "").strip()

# Per-endpoint rate limiting
RATE_LIMIT_ENABLED = _env_bool("RATE_LIMIT_ENABLED", True)
RATE_LIMIT_WINDOW_SEC = int((os.getenv("RATE_LIMIT_WINDOW_SEC") or "60").strip())
RATE_LIMIT_QUERY_MAX = int((os.getenv("RATE_LIMIT_QUERY_MAX") or "30").strip())
RATE_LIMIT_CHAT_MAX = int((os.getenv("RATE_LIMIT_CHAT_MAX") or "60").strip())
RATE_LIMIT_WS_MAX = int((os.getenv("RATE_LIMIT_WS_MAX") or "30").strip())

# Auth/tenant fallback policy
# WARNING: allowing uid-as-tenant fallback defeats tenant isolation.
# Enable ONLY for single-tenant dev/testing.
ALLOW_UID_AS_TENANT_FALLBACK = _env_bool("ALLOW_UID_AS_TENANT_FALLBACK", False)

if IS_PRODUCTION and ALLOW_UID_AS_TENANT_FALLBACK:
    import warnings
    warnings.warn(
        "ALLOW_UID_AS_TENANT_FALLBACK is enabled in production! "
        "This defeats tenant isolation. Set ALLOW_UID_AS_TENANT_FALLBACK=0.",
        stacklevel=1,
    )

# Prompt templates
SYSTEM_PROMPT_PATH = str(PROJECT_ROOT / "app" / "resources" / "prompts" / "system_vi.md")

# Smalltalk / cheap routing (avoid LLM calls)
SMALLTALK_PATH = str(PROJECT_ROOT / "app" / "resources" / "smalltalk_vi.json")
ENABLE_SMALLTALK = True
SMALLTALK_COSINE_THRESHOLD = 0.78

# Agentic router (Day 4)
ENABLE_SEMANTIC_ROUTER = True
TOXIC_MESSAGE = "Dạ em xin phép không hỗ trợ nội dung này. Anh/chị cần tư vấn khóa học/học phí/lịch học nào của trung tâm ạ?"

# In-domain anchors (cheap pre-check before retrieval)
DOMAIN_ANCHORS_PATH = str(PROJECT_ROOT / "app" / "resources" / "domain_anchors_vi.json")
DOMAIN_ANCHOR_COSINE_THRESHOLD = 0.25
DOMAIN_KEYWORDS = [
    # Vietnamese (with/without accents are handled by code)
    "trung tâm",
    "trung tam",
    "ở đâu",
    "o dau",
    "địa điểm",
    "dia diem",
    "học phí",
    "hoc phi",
    "lịch học",
    "lich hoc",
    "khai giảng",
    "khai giang",
    "ưu đãi",
    "uu dai",
    "địa chỉ",
    "dia chi",
    "cơ sở",
    "co so",
    "chi nhánh",
    "chi nhanh",
    "giao tiếp",
    "giao tiep",
    "ielts",
    "toeic",
    "thiếu nhi",
    "thieu nhi",
    "đầu vào",
    "dau vao",
    "xếp lớp",
    "xep lop",
    "lộ trình",
    "lo trinh",
    "giáo viên",
    "giao vien",
]

# Qdrant connection and collection
QDRANT_HOST = (os.getenv("QDRANT_HOST") or "localhost").strip()
QDRANT_PORT = int((os.getenv("QDRANT_PORT") or "6333").strip())
COLLECTION_NAME = (os.getenv("QDRANT_COLLECTION") or os.getenv("COLLECTION_NAME") or "RAG_docs").strip()

# Multi-tenant / multi-branch isolation
TENANT_FIELD = "tenant_id"
BRANCH_FIELD = "branch_id"
ENABLE_BRANCH_FILTER = True
# In CV_MODE, allow cross-category retrieval if tenant_id is None
REQUIRE_TENANT_ID = _env_bool("REQUIRE_TENANT_ID", not CV_MODE)
ENFORCE_METADATA_FILTERS = _env_bool("ENFORCE_METADATA_FILTERS", True)

# Vector settings (must match embedding model dimension)
VECTOR_SIZE = 1024
VECTOR_DISTANCE = Distance.COSINE

# Embedding model
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# Data locations
DATA_PATH = str(PROJECT_ROOT / "data" / "knowledge_base")
NODES_CACHE_PATH = str(PROJECT_ROOT / "data" / ".cache" / "nodes.jsonl")

# Chunking (shared for ingest and BM25 fallback)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# In-Context RALM
FEWSHOT_PATH = str(PROJECT_ROOT / "app" / "resources" / "eval" / "fewshot_examples.json")
RETRIEVAL_TOP_K = 5 
EXAMPLES_TOP_K = 3

# BM25 / Hybrid retrieval
USE_BM25 = True # bật tắt hybrid/lexcial search 
BM25_SOURCE = "nodes_file"  # options: "nodes_file", "files"
BM25_TOP_K = 5
BM25_MAX_CHARS = 800  # used only in legacy files mode
BM25_K1 = 1.5
BM25_B = 0.75
HYBRID_ALPHA = 0.5  # 1.0 = vector-only, 0.0 = BM25-only

# Debug/trace controls (default OFF to avoid leaking prompts/PII into logs)
DEBUG_VERBOSE = (os.getenv("DEBUG_VERBOSE") or "0").strip().lower() in ("1", "true", "yes", "on")
DEBUG_TOPN_PRINT = int(os.getenv("DEBUG_TOPN_PRINT") or "3")
DEBUG_SHOW_PROMPT = (os.getenv("DEBUG_SHOW_PROMPT") or "0").strip().lower() in ("1", "true", "yes", "on")

# Postgres (Day 6-7 persistent memory)
# Prefer configuring via env/.env; keep code default empty to avoid hardcoding credentials.
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
CHAT_SESSIONS_TABLE = (os.getenv("CHAT_SESSIONS_TABLE") or "chat_sessions").strip()

# Memory controls (Day 6-7)
MEMORY_ENABLED = (os.getenv("MEMORY_ENABLED") or "1").strip().lower() in ("1", "true", "yes", "on")
MEMORY_LAST_TURNS = int(os.getenv("MEMORY_LAST_TURNS") or "6")  # 6 turns = 12 messages (user+assistant)
MEMORY_BUDGET_TOKENS = int(os.getenv("MEMORY_BUDGET_TOKENS") or "1000")  # apply to (summary + last N turns)
MEMORY_SUMMARY_ENABLED = (os.getenv("MEMORY_SUMMARY_ENABLED") or "1").strip().lower() in ("1", "true", "yes", "on")
MEMORY_SUMMARY_MAX_OUTPUT_TOKENS = int(os.getenv("MEMORY_SUMMARY_MAX_OUTPUT_TOKENS") or "350")

# Long-term memory controls
MEMORY_LONGTERM_MIN_CONFIDENCE = float(os.getenv("MEMORY_LONGTERM_MIN_CONFIDENCE") or "0.7")
MEMORY_LONGTERM_MAX_FACTS = int(os.getenv("MEMORY_LONGTERM_MAX_FACTS") or "8")

# Heuristic extraction (Hook 2A)
MEMORY_HEURISTIC_ENABLED = (os.getenv("MEMORY_HEURISTIC_ENABLED") or "1").strip().lower() in ("1", "true", "yes", "on")
MEMORY_HEURISTIC_TOOL_CONFIDENCE = float(os.getenv("MEMORY_HEURISTIC_TOOL_CONFIDENCE") or "0.85")
MEMORY_HEURISTIC_REGEX_CONFIDENCE = float(os.getenv("MEMORY_HEURISTIC_REGEX_CONFIDENCE") or "0.4")
MEMORY_HEURISTIC_MAX_MESSAGES = int(os.getenv("MEMORY_HEURISTIC_MAX_MESSAGES") or "12")
MEMORY_HEURISTIC_DEDUPE_TTL_SEC = int(os.getenv("MEMORY_HEURISTIC_DEDUPE_TTL_SEC") or "21600")
MEMORY_HEURISTIC_WORKERS = int(os.getenv("MEMORY_HEURISTIC_WORKERS") or "2")

# LLM verification (Hook 2B)
MEMORY_LLM_VERIFY_ENABLED = (os.getenv("MEMORY_LLM_VERIFY_ENABLED") or "1").strip().lower() in ("1", "true", "yes", "on")
MEMORY_LLM_CONFIRM_THRESHOLD = float(os.getenv("MEMORY_LLM_CONFIRM_THRESHOLD") or "0.8")
MEMORY_LLM_VERIFY_MAX_CANDIDATES = int(os.getenv("MEMORY_LLM_VERIFY_MAX_CANDIDATES") or "8")
MEMORY_LLM_VERIFY_MAX_MESSAGES = int(os.getenv("MEMORY_LLM_VERIFY_MAX_MESSAGES") or "12")
MEMORY_LLM_CONFLICT_MARGIN = float(os.getenv("MEMORY_LLM_CONFLICT_MARGIN") or "0.15")
MEMORY_LLM_ASK_ON_CONFLICT = (os.getenv("MEMORY_LLM_ASK_ON_CONFLICT") or "1").strip().lower() in ("1", "true", "yes", "on")

# Prompt budget controls
MAX_PROMPT_CHARS = 9000
PER_CHUNK_PROMPT_MAX_CHARS = 1000
PROMPT_TOP_CONTEXTS = 3  # clamp exact top-N contexts sent to LLM after rerank

# Conversation memory (stateful)
HISTORY_ENABLED = True
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS") or "12")  # default ~6 turns (user+assistant)
HISTORY_MSG_MAX_CHARS = 300      # cắt mỗi message để giữ prompt gọn

# Reranking controls
RERANK_USE_COSINE = True
RERANK_TOP_M = 20
RERANK_WEIGHT = 0.5  # combine fused score and cosine (0..1)

# In-domain / out-of-domain guard (avoid LLM calls on low-confidence retrieval)
ENABLE_DOMAIN_GUARD = True
# cosine(query, best_chunk_text) below this → treat as out-of-domain
DOMAIN_COSINE_THRESHOLD = 0.33
OUT_OF_DOMAIN_MESSAGE = (
    "Dạ em chỉ hỗ trợ tư vấn các thông tin liên quan đến trung tâm (khóa học, học phí, lịch học, ưu đãi...). "
    "Nếu anh/chị cho em xin **SĐT** và **nhu cầu học**, em sẽ chuyển tư vấn viên liên hệ hỗ trợ chi tiết hơn ạ."
)

# In-domain but not enough evidence in the tenant knowledge base.
NO_MATCH_MESSAGE = (
    "Dạ hiện tại em **chưa tìm thấy thông tin phù hợp trong tài liệu của trung tâm** để trả lời câu này. "
    "Anh/chị có thể hỏi cụ thể hơn (tên khóa/IELTS/TOEIC/lịch học/học phí/ưu đãi...) giúp em được không ạ? "
    "Nếu anh/chị cho em xin **SĐT** và **nhu cầu học**, em sẽ chuyển tư vấn viên hỗ trợ chi tiết hơn ạ."
)

# LLM reliability
LLM_MAX_RETRIES = 3
LLM_RETRY_INITIAL_DELAY = 1.0
LLM_RETRY_BACKOFF = 2.0
LLM_429_SLEEP_SECS = 20
LLM_429_JITTER_SECS = 10

# LLM fallback behavior (when quota/network/provider errors happen)
LLM_FALLBACK_TO_CONTEXT_ON_ERROR = True
LLM_FALLBACK_CONTEXT_SNIPPET_CHARS = 1200
