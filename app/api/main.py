import asyncio
import logging
import os
import time
import threading
from typing import Any, Dict, List, Optional, cast

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.core.bootstrap import bootstrap_runtime
from app.core.config import PROJECT_ROOT
from app.api.deps import get_current_user
from app.api.owner_console import require_owner, router as owner_router
from app.services.agentic.service import semantic_router_response
from app.services.rag_service import build_index, rag_query
from app.services.analytics.store import (
    insert_feedback,
    insert_trace,
    list_handoffs,
    list_tenants,
    list_traces,
    metrics,
    new_trace_id,
)
from app.core.config import (
    ADMIN_REQUIRE_OWNER_AUTH,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_ORIGINS,
    ENABLE_BRANCH_FILTER,
    ENABLE_PUBLIC_QUERY_ENDPOINT,
    PUBLIC_QUERY_API_KEY,
    RATE_LIMIT_CHAT_MAX,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_QUERY_MAX,
    RATE_LIMIT_WINDOW_SEC,
    RATE_LIMIT_WS_MAX,
)
from app.core.rate_limit import InMemorySlidingWindowRateLimiter


logger = logging.getLogger(__name__)

app = FastAPI(title="Viet RAG2 Multitenant Assistant")
app.include_router(owner_router)

# CORS policy is configured via environment variables.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

_RATE_LIMITER = InMemorySlidingWindowRateLimiter()


class QueryRequest(BaseModel):
    question: str
    tenant_id: Optional[str] = None
    branch_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class SemanticRequest(BaseModel):
    question: str
    tenant_id: Optional[str] = None
    branch_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    user_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    trace_id: Optional[str] = None
    time_ms: Optional[float] = None
    route: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    branch_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    sources: List[str]
    tenant_used: str
    trace_id: Optional[str] = None
    time_ms: Optional[float] = None
    route: Optional[str] = None


class FeedbackRequest(BaseModel):
    trace_id: str
    tenant_id: Optional[str] = None
    rating: int  # 1=up, -1=down
    comment: Optional[str] = None


def _client_ip_from_request(request: Request) -> str:
    xff = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if xff:
        return xff
    if request.client and request.client.host:
        return str(request.client.host)
    return "unknown"


def _client_ip_from_ws(ws: WebSocket) -> str:
    xff = (ws.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if xff:
        return xff
    if ws.client and ws.client.host:
        return str(ws.client.host)
    return "unknown"


def _bearer_creds_from_header(auth_header: Optional[str]) -> Optional[HTTPAuthorizationCredentials]:
    raw = (auth_header or "").strip()
    if not raw.lower().startswith("bearer "):
        return None
    token = raw[7:].strip()
    if not token:
        return None
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _enforce_http_rate_limit(scope: str, request: Request, *, principal: Optional[str] = None) -> None:
    if not RATE_LIMIT_ENABLED:
        return
    limit = RATE_LIMIT_QUERY_MAX if scope == "query" else RATE_LIMIT_CHAT_MAX
    key = f"{scope}:{principal or _client_ip_from_request(request)}"
    allowed, retry_after = _RATE_LIMITER.allow(key=key, limit=limit, window_sec=RATE_LIMIT_WINDOW_SEC)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please retry later.",
            headers={"Retry-After": str(retry_after)},
        )


def _enforce_ws_rate_limit(ws: WebSocket, *, principal: Optional[str] = None) -> tuple[bool, int]:
    if not RATE_LIMIT_ENABLED:
        return True, 0
    key = f"ws_query:{principal or _client_ip_from_ws(ws)}"
    return _RATE_LIMITER.allow(key=key, limit=RATE_LIMIT_WS_MAX, window_sec=RATE_LIMIT_WINDOW_SEC)


def _admin_guard(request: Request) -> Dict[str, Any]:
    """Always require owner JWT for /admin/api/* endpoints."""
    creds = _bearer_creds_from_header(request.headers.get("authorization"))
    return require_owner(request, creds)


def _guard_public_query_access(request: Request) -> None:
    if ENABLE_PUBLIC_QUERY_ENDPOINT:
        return

    # Optional API key path (for server-to-server callers).
    if PUBLIC_QUERY_API_KEY:
        req_key = (request.headers.get("x-api-key") or "").strip()
        if req_key and req_key == PUBLIC_QUERY_API_KEY:
            return

    # Owner is always allowed when owner auth is enabled.
    if ADMIN_REQUIRE_OWNER_AUTH:
        try:
            _ = _admin_guard(request)
            return
        except HTTPException:
            pass

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Public /query endpoint is disabled. Use /chat with tenant auth.",
    )


def _guard_public_ws_access(ws: WebSocket) -> tuple[bool, str]:
    if ENABLE_PUBLIC_QUERY_ENDPOINT:
        return True, ""

    if PUBLIC_QUERY_API_KEY:
        req_key = (ws.headers.get("x-api-key") or "").strip()
        if req_key and req_key == PUBLIC_QUERY_API_KEY:
            return True, ""

    if ADMIN_REQUIRE_OWNER_AUTH:
        try:
            creds = _bearer_creds_from_header(ws.headers.get("authorization"))
            _ = require_owner(cast(Request, ws), creds)
            return True, ""
        except HTTPException:
            pass

    return False, "Public websocket query is disabled. Use authenticated /chat endpoint."


def _parse_date_like(s: Optional[str]) -> Optional[int]:
    """
    Parse YYYY-MM-DD or epoch seconds to int epoch seconds.
    """
    if not s:
        return None
    raw = str(s).strip()
    if not raw:
        return None
    if raw.isdigit():
        try:
            return int(raw)
        except Exception:
            return None
    try:
        # YYYY-MM-DD
        parts = raw.split("T", 1)[0].split("-")
        if len(parts) == 3:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            import datetime as _dt

            return int(_dt.datetime(y, m, d).timestamp())
    except Exception:
        return None
    return None


_READY = threading.Event()  # set once models are loaded


def _background_init():
    """Load models in background so server can start accepting connections immediately."""
    try:
        bootstrap_runtime()
        build_index()
        logger.info("✅ RAG backend is ready (models loaded).")
    except Exception as e:
        logger.error("❌ Background init failed: %s", e)
    finally:
        _READY.set()


@app.on_event("startup")
def startup_event() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logger.info("Starting RAG backend...")
    if CORS_ALLOW_ORIGINS == ["*"]:
        logger.warning("⚠️  CORS origins is set to wildcard (*). Not recommended for production.")
    elif not CORS_ALLOW_ORIGINS:
        logger.info("CORS: no origins allowed (same-origin only).")
    else:
        logger.info("CORS: allowed origins = %s", CORS_ALLOW_ORIGINS)
    rag_init = (os.getenv("RAG_INIT_ON_STARTUP") or "1").strip().lower() in ("1", "true", "yes", "on")
    if rag_init:
        t = threading.Thread(target=_background_init, daemon=True)
        t.start()
        logger.info("⏳ Models loading in background thread... Server is accepting connections.")
    else:
        _READY.set()
        logger.info("Skipping RAG init on startup (RAG_INIT_ON_STARTUP=0).")


@app.get("/health")
def health() -> Dict[str, str]:
    ready = _READY.is_set()
    return {"status": "ok" if ready else "loading", "models_ready": str(ready).lower()}

@app.get("/public/config")
def public_config() -> Dict[str, object]:
    return {
        "enable_branch_filter": bool(ENABLE_BRANCH_FILTER),
        "public_query_enabled": bool(ENABLE_PUBLIC_QUERY_ENDPOINT),
    }

@app.post("/semantic", include_in_schema=True)
def semantic_endpoint(payload: SemanticRequest, request: Request):
    """
    Semantic Router output mode (per prompt spec):
    - tool needed -> pure JSON {tool_name, arguments, thought}
    - otherwise -> plain text
    """
    # Fast path: try semantic routing without bootstrapping embeddings/LLM/Qdrant.
    # This covers calculator/comparison/ticket intents and keeps `/semantic` responsive even when vector DB is down.
    _guard_public_query_access(request)
    _enforce_http_rate_limit("query", request, principal=payload.user_id)
    try:
        out = semantic_router_response(
            payload.question,
            index=None,
            tenant_id=payload.tenant_id,
            branch_id=payload.branch_id,
            history=payload.history or [],
            user_id=payload.user_id,
        )
    except RuntimeError:
        # Course search requires RAG (embeddings + Qdrant index).
        try:
            bootstrap_runtime()
            index = build_index()
            out = semantic_router_response(
                payload.question,
                index=index,
                tenant_id=payload.tenant_id,
                branch_id=payload.branch_id,
                history=payload.history or [],
                user_id=payload.user_id,
            )
        except Exception as e:
            return PlainTextResponse(str(e), status_code=503)
    if isinstance(out, dict):
        return JSONResponse(out)
    return PlainTextResponse(str(out or ""))

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/agent")


WEB_DIR = PROJECT_ROOT / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/admin", include_in_schema=False)
def admin_ui() -> Response:
    return RedirectResponse(url="/owner")


@app.get("/agent", include_in_schema=False)
def agent_ui() -> FileResponse:
    return FileResponse(str(WEB_DIR / "agent.html"))


@app.get("/admin/api/tenants")
def admin_tenants(_owner: Dict[str, Any] = Depends(_admin_guard)) -> Dict[str, object]:
    return {"tenants": list_tenants()}


@app.get("/admin/api/metrics")
def admin_metrics(
    tenant_id: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    _owner: Dict[str, Any] = Depends(_admin_guard),
) -> Dict[str, object]:
    return metrics(tenant_id=tenant_id, since_ts=_parse_date_like(since), until_ts=_parse_date_like(until))


@app.get("/admin/api/logs")
def admin_logs(
    tenant_id: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    route: Optional[str] = None,
    status: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    _owner: Dict[str, Any] = Depends(_admin_guard),
) -> Dict[str, object]:
    rows = list_traces(
        tenant_id=tenant_id,
        since_ts=_parse_date_like(since),
        until_ts=_parse_date_like(until),
        route=route,
        status=status,
        q=q,
        limit=limit,
        offset=offset,
    )
    return {"rows": rows}


@app.get("/admin/api/handoffs")
def admin_handoffs(
    tenant_id: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    _owner: Dict[str, Any] = Depends(_admin_guard),
) -> Dict[str, object]:
    rows = list_handoffs(
        tenant_id=tenant_id,
        since_ts=_parse_date_like(since),
        until_ts=_parse_date_like(until),
        status=status,
        limit=limit,
        offset=offset,
    )
    return {"rows": rows}


@app.post("/admin/api/feedback")
def admin_feedback(payload: FeedbackRequest, _owner: Dict[str, Any] = Depends(_admin_guard)) -> Dict[str, object]:
    rating = 1 if int(payload.rating) > 0 else -1
    fb_id = insert_feedback(trace_id=payload.trace_id, tenant_id=payload.tenant_id, rating=rating, comment=payload.comment)
    return {"ok": True, "id": fb_id}


@app.post("/feedback")
def public_feedback(payload: FeedbackRequest, request: Request) -> Dict[str, object]:
    _enforce_http_rate_limit("query", request, principal=f"feedback:{_client_ip_from_request(request)}")
    rating = 1 if int(payload.rating) > 0 else -1
    fb_id = insert_feedback(trace_id=payload.trace_id, tenant_id=payload.tenant_id, rating=rating, comment=payload.comment)
    return {"ok": True, "id": fb_id}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest, request: Request) -> QueryResponse:
    _guard_public_query_access(request)
    _enforce_http_rate_limit("query", request, principal=payload.user_id)
    if not _READY.wait(timeout=300):
        raise HTTPException(status_code=503, detail="Models are still loading. Please try again later.")

    trace_id = new_trace_id()
    start = time.perf_counter()
    result = None
    err = None
    try:
        result = rag_query(
            question=payload.question,
            tenant_id=payload.tenant_id,
            branch_id=payload.branch_id,
            history=payload.history or [],
            channel="web",
            session_id=payload.session_id,
            user_id=payload.user_id,
        )
    except Exception as e:
        err = str(e)
        result = {"answer": "", "sources": [], "route": "error"}
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    sources = (result.get("sources", []) or []) if isinstance(result, dict) else []
    route = (result.get("route") if isinstance(result, dict) else None) or None
    tool_md = (result.get("tool_metadata") if isinstance(result, dict) else None) or {}
    status = "ERROR" if err else "SUCCESS"
    logger.info(
        "query tenant=%s branch=%s len_q=%d time_ms=%.1f sources=%d",
        payload.tenant_id or "-",
        payload.branch_id or "-",
        len(payload.question or ""),
        elapsed_ms,
        len(sources),
    )
    try:
        insert_trace(
            trace_id=trace_id,
            tenant_id=payload.tenant_id,
            branch_id=payload.branch_id,
            channel="web",
            session_id=payload.session_id,
            user_id=payload.user_id,
            question=payload.question or "",
            answer=str(result.get("answer", "") if isinstance(result, dict) else ""),
            sources=[str(s) for s in sources],
            route=str(route) if route else None,
            status=status,
            latency_ms=float(elapsed_ms),
            tool_metadata=tool_md if isinstance(tool_md, dict) else {},
            error=err,
        )
    except Exception as e:
        logger.warning("Failed to insert trace: %s", e)
    return QueryResponse(
        answer=str(result.get("answer", "") if isinstance(result, dict) else ""),
        sources=[str(s) for s in sources],
        trace_id=trace_id,
        time_ms=round(elapsed_ms, 1),
        route=str(route) if route else None,
    )


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest, request: Request, current_user: dict = Depends(get_current_user)) -> ChatResponse:
    """
    Protected chat endpoint:
    - Requires Firebase ID token (Authorization: Bearer <id_token>)
    - Uses tenant_id from decoded token (custom claim) to enforce tenant isolation
    """
    tenant_id = str(current_user.get("tenant_id") or "").strip()
    user_id = str(current_user.get("uid") or "").strip() or None
    if not tenant_id:
        # Fail-closed: no tenant => do not serve chat.
        raise ValueError("Missing tenant_id in token.")

    _enforce_http_rate_limit("chat", request, principal=user_id or tenant_id)

    session_id = (payload.session_id or "").strip() or f"{tenant_id}:web:{user_id or 'user'}"

    trace_id = new_trace_id()
    start = time.perf_counter()
    err = None
    try:
        result = rag_query(
            question=payload.message,
            tenant_id=tenant_id,
            branch_id=payload.branch_id,
            history=[],
            channel="tenant_chat",
            user_id=user_id,
            session_id=session_id,
        )
    except Exception as e:
        err = str(e)
        result = {"answer": "", "sources": [], "route": "error"}

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    sources = (result.get("sources", []) or []) if isinstance(result, dict) else []
    route = (result.get("route") if isinstance(result, dict) else None) or None
    tool_md = (result.get("tool_metadata") if isinstance(result, dict) else None) or {}

    try:
        insert_trace(
            trace_id=trace_id,
            tenant_id=tenant_id,
            branch_id=payload.branch_id,
            channel="tenant_chat",
            session_id=session_id,
            user_id=user_id,
            question=payload.message or "",
            answer=str(result.get("answer", "") if isinstance(result, dict) else ""),
            sources=[str(s) for s in sources],
            route=str(route) if route else None,
            status=("ERROR" if err else "SUCCESS"),
            latency_ms=float(elapsed_ms),
            tool_metadata=tool_md if isinstance(tool_md, dict) else {},
            error=err,
        )
    except Exception as e:
        logger.warning("Failed to insert trace (chat): %s", e)

    return ChatResponse(
        reply=str(result.get("answer", "") if isinstance(result, dict) else ""),
        sources=[str(s) for s in sources],
        tenant_used=tenant_id,
        trace_id=trace_id,
        time_ms=round(elapsed_ms, 1),
        route=str(route) if route else None,
    )


@app.websocket("/ws/query")
async def websocket_query(ws: WebSocket) -> None:
    """
    WebSocket endpoint to stream incremental answers to frontend.

    Protocol:
    - Client sends JSON: {"question": str, "tenant_id": str | null, "history": [...]}
    - Server responds in sequence:
        {"type": "meta", "sources": [...], "time_ms": float}
        {"type": "chunk", "text": "..."}  (repeated multiple times)
        {"type": "end"}
      Or on error:
        {"type": "error", "message": "..."}
    """
    await ws.accept()
    ok, msg = _guard_public_ws_access(ws)
    if not ok:
        await ws.send_json({"type": "error", "message": msg})
        await ws.close(code=1008)
        return

    # Wait for models to be loaded (background init)
    if not _READY.is_set():
        await ws.send_json({"type": "chunk", "text": "⏳ Hệ thống đang khởi tạo, vui lòng chờ..."})
        loop = asyncio.get_running_loop()
        ready = await loop.run_in_executor(None, lambda: _READY.wait(timeout=300))
        if not ready:
            await ws.send_json({"type": "error", "message": "Server chưa sẵn sàng. Vui lòng thử lại sau."})
            await ws.close(code=1013)
            return

    logger.info("WebSocket connected")
    try:
        while True:
            data = await ws.receive_json()
            question = (data.get("question") or "").strip()
            tenant_id = data.get("tenant_id")
            branch_id = data.get("branch_id")
            session_id = data.get("session_id")
            user_id = data.get("user_id")
            history = data.get("history") or []
            if not question:
                await ws.send_json({"type": "error", "message": "Câu hỏi trống."})
                continue

            principal = str(user_id or session_id or tenant_id or _client_ip_from_ws(ws))
            allowed, retry_after = _enforce_ws_rate_limit(ws, principal=principal)
            if not allowed:
                await ws.send_json(
                    {
                        "type": "error",
                        "message": f"Too many websocket requests. Retry after {retry_after}s.",
                    }
                )
                continue

            start = time.perf_counter()
            loop = asyncio.get_running_loop()
            trace_id = new_trace_id()
            err = None
            gen = await loop.run_in_executor(
                None,
                lambda: rag_query(
                    question=question,
                    tenant_id=tenant_id,
                    branch_id=branch_id,
                    history=history,
                    channel="web",
                    session_id=session_id,
                    user_id=user_id,
                    stream=True,
                ),
            )

            answer = ""
            sources = []
            route = None
            tool_md = {}
            elapsed_ms = 0.0
            
            while True:
                try:
                    item = await loop.run_in_executor(None, next, gen)
                except StopIteration:
                    break
                except Exception as e:
                    logger.exception("Error in stream generator: %s", e)
                    await ws.send_json({"type": "error", "message": "Lỗi sinh luồng dữ liệu phản hồi."})
                    break

                if item["type"] == "meta":
                    sources = item.get("data", {}).get("sources", [])
                    route = item.get("data", {}).get("route")
                    await ws.send_json({
                        "type": "meta",
                        "time_ms": round((time.perf_counter() - start) * 1000.0, 1),
                        "sources": [str(s) for s in sources],
                        "trace_id": trace_id,
                        "route": str(route) if route else None,
                    })
                elif item["type"] == "chunk":
                    chunk_text = item.get("content", "")
                    if chunk_text:
                        answer += chunk_text
                        await ws.send_json({"type": "chunk", "text": chunk_text})
                elif item["type"] == "result":
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    res_data = item.get("data", {})
                    answer = str(res_data.get("answer", "") or "")
                    sources = res_data.get("sources", []) or []
                    route = (res_data.get("route") if isinstance(res_data, dict) else None) or None
                    tool_md = (res_data.get("tool_metadata") if isinstance(res_data, dict) else None) or {}
                    
                    logger.info(
                        "ws_query tenant=%s branch=%s len_q=%d time_ms=%.1f sources=%d",
                        tenant_id or "-",
                        branch_id or "-",
                        len(question),
                        elapsed_ms,
                        len(sources),
                    )

                    try:
                        insert_trace(
                            trace_id=trace_id,
                            tenant_id=tenant_id,
                            branch_id=branch_id,
                            channel="web_ws",
                            session_id=session_id,
                            user_id=user_id,
                            question=question,
                            answer=answer,
                            sources=[str(s) for s in sources],
                            route=str(route) if route else None,
                            status="SUCCESS",
                            latency_ms=float(elapsed_ms),
                            tool_metadata=tool_md if isinstance(tool_md, dict) else {},
                        )
                    except Exception as e:
                        logger.warning("Failed to insert ws trace: %s", e)

            if not answer:
                await ws.send_json({"type": "chunk", "text": "(Không có câu trả lời) "})

            await ws.send_json({"type": "end"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ============================================================
# CV Upload & Review Endpoints
# ============================================================

_CV_MAX_FILES = 10
_CV_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
_CV_ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".md"}


class CVReviewRequest(BaseModel):
    candidate_id: Optional[str] = None
    cv_text: str
    job_description: Optional[str] = None


def _validate_cv_upload(files: List[UploadFile]) -> None:
    """Validate uploaded files before processing."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > _CV_MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {_CV_MAX_FILES} files per upload.",
        )
    for f in files:
        name = (f.filename or "unknown").lower()
        import os as _os
        ext = _os.path.splitext(name)[1]
        if ext not in _CV_ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Allowed: {', '.join(_CV_ALLOWED_EXTENSIONS)}",
            )


@app.post("/cv/upload")
async def cv_upload(
    request: Request,
    files: List[UploadFile] = File(...),
    job_description: Optional[str] = None,
) -> JSONResponse:
    """
    Upload CV files (PDF/DOCX/TXT) for batch scoring.

    Returns a sorted list of candidates with scores.
    Max 10 files, 10MB each.
    """
    _guard_public_query_access(request)
    _enforce_http_rate_limit("query", request)
    _validate_cv_upload(files)

    from app.services.cv_parser import extract_text, guess_candidate_name
    from app.services.cv_scorer import quick_score_batch

    cv_texts: list[tuple[str, str]] = []  # [(filename, text), ...]
    errors: list[dict] = []

    for f in files:
        filename = f.filename or "unknown"
        try:
            file_bytes = await f.read()

            # File size check
            if len(file_bytes) > _CV_MAX_FILE_SIZE:
                errors.append({"file": filename, "error": f"File too large (max {_CV_MAX_FILE_SIZE // 1024 // 1024}MB)"})
                continue

            text = extract_text(file_bytes, filename)
            if not text.strip():
                errors.append({"file": filename, "error": "Could not extract text from file"})
                continue

            cv_texts.append((filename, text))
        except Exception as e:
            logger.error("Failed to process uploaded file %s: %s", filename, e)
            errors.append({"file": filename, "error": str(e)})

    if not cv_texts:
        raise HTTPException(
            status_code=422,
            detail={"message": "No valid CV content could be extracted.", "errors": errors},
        )

    jd_text = (job_description or "").strip()
    candidates = quick_score_batch(cv_texts, jd_text)

    result = {
        "candidates": [c.model_dump() for c in candidates],
    }
    if errors:
        result["errors"] = errors

    return JSONResponse(content=result)


@app.post("/cv/review")
async def cv_review(
    payload: CVReviewRequest,
    request: Request,
) -> JSONResponse:
    """
    Detailed review of a single CV.

    Returns structured evaluation with score, strengths, weaknesses,
    reasoning, and phrase-level annotations.
    """
    _guard_public_query_access(request)
    _enforce_http_rate_limit("query", request)

    if not payload.cv_text.strip():
        raise HTTPException(status_code=400, detail="cv_text is required.")

    from app.services.cv_scorer import score_cv

    jd_text = (payload.job_description or "").strip()
    review = score_cv(payload.cv_text, jd_text)

    return JSONResponse(content={
        "candidate_id": payload.candidate_id,
        "score": review.score,
        "strengths": review.strengths,
        "weaknesses": review.weaknesses,
        "reasoning": review.reasoning,
        "annotations": [a.model_dump() for a in review.annotations],
    })
