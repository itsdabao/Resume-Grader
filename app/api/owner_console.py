from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import PROJECT_ROOT
from app.services.analytics.store import list_handoffs, list_tenants, list_traces, metrics


router = APIRouter()

WEB_DIR = PROJECT_ROOT / "web"
OWNER_COOKIE_NAME = "owner_token"


def _parse_date_like(s: Optional[str]) -> Optional[int]:
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
        parts = raw.split("T", 1)[0].split("-")
        if len(parts) == 3:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            import datetime as _dt

            return int(_dt.datetime(y, m, d).timestamp())
    except Exception:
        return None
    return None


def _require_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing env var: {name}. Please set it in `.env` (or OS env) and restart the server.",
        )
    return v


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    s = (data or "").strip()
    if not s:
        raise ValueError("empty base64url")
    s += "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s.encode("ascii"))


def _jwt_encode(payload: Dict[str, Any], *, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(sig)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def _jwt_decode_verify(token: str, *, secret: str) -> Dict[str, Any]:
    parts = (token or "").split(".")
    if len(parts) != 3:
        raise ValueError("invalid jwt format")
    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    got_sig = _b64url_decode(sig_b64)
    if not hmac.compare_digest(expected_sig, got_sig):
        raise ValueError("invalid jwt signature")
    header = json.loads(_b64url_decode(header_b64).decode("utf-8"))
    if (header.get("alg") or "").upper() != "HS256":
        raise ValueError("unsupported jwt alg")
    payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and int(exp) < int(time.time()):
        raise ValueError("jwt expired")
    return payload


bearer = HTTPBearer(auto_error=False)


def require_owner(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> Dict[str, Any]:
    secret = _require_env("JWT_SECRET")

    token = None
    if creds and (creds.credentials or "").strip():
        token = creds.credentials.strip()
    if not token:
        token = (request.cookies.get(OWNER_COOKIE_NAME) or "").strip() or None
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing owner token")

    try:
        payload = _jwt_decode_verify(token, secret=secret)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired owner token")

    if str(payload.get("type") or "") != "owner":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
    sub = str(payload.get("sub") or "").strip()
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject")
    return {"username": sub, "claims": payload}


@router.get("/owner", include_in_schema=False)
def owner_ui() -> FileResponse:
    path = WEB_DIR / "owner.html"
    if not path.exists():
        raise HTTPException(status_code=500, detail="Missing web/owner.html")
    return FileResponse(str(path))


@router.post("/owner/auth/login")
async def owner_login(payload: Dict[str, Any], response: Response) -> Dict[str, Any]:
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "").strip()
    expected_u = _require_env("OWNER_USERNAME")
    expected_p = _require_env("OWNER_PASSWORD")
    if username != expected_u or password != expected_p:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    raw_exp = (os.getenv("JWT_EXPIRE_MIN") or "1440").strip() or "1440"
    try:
        expire_min = int(raw_exp)
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid env var JWT_EXPIRE_MIN")
    now = int(time.time())
    exp = now + max(1, expire_min) * 60
    token = _jwt_encode({"type": "owner", "sub": username, "iat": now, "exp": exp}, secret=_require_env("JWT_SECRET"))

    # Local-first: keep Secure=False by default so it works on http://localhost
    response.set_cookie(
        key=OWNER_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
        max_age=max(1, exp - now),
    )
    return {"ok": True}


@router.post("/owner/auth/logout")
async def owner_logout(response: Response) -> Dict[str, Any]:
    response.delete_cookie(key=OWNER_COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/owner/api/tenants")
def owner_tenants(_owner: Dict[str, Any] = Depends(require_owner)) -> Dict[str, Any]:
    return {"tenants": list_tenants()}


@router.get("/owner/api/metrics")
def owner_metrics(
    tenant_id: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    _owner: Dict[str, Any] = Depends(require_owner),
) -> Dict[str, Any]:
    return metrics(tenant_id=tenant_id, since_ts=_parse_date_like(since), until_ts=_parse_date_like(until))


@router.get("/owner/api/logs")
def owner_logs(
    tenant_id: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    route: Optional[str] = None,
    status: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    _owner: Dict[str, Any] = Depends(require_owner),
) -> Dict[str, Any]:
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
    out = []
    for r in rows:
        q_preview = str(r.get("question") or "").replace("\n", " ").strip()
        out.append(
            {
                "trace_id": r.get("trace_id"),
                "ts": r.get("ts"),
                "tenant_id": r.get("tenant_id"),
                "route": r.get("route"),
                "status": r.get("status"),
                "latency_ms": r.get("latency_ms"),
                "sources_count": r.get("sources_count"),
                "question_preview": q_preview[:280],
            }
        )
    return {"rows": out}


@router.get("/owner/api/logs/{trace_id}")
def owner_log_detail(trace_id: str, _owner: Dict[str, Any] = Depends(require_owner)) -> Dict[str, Any]:
    # Avoid importing SQLAlchemy models here; re-use the existing list endpoint by filtering in DB.
    # We use qdrant_id-style trace_id (uuid hex), so a direct select is safe and cheap.
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from app.services.analytics.store import RequestTrace, ensure_analytics_tables_exist
    from app.services.memory.store import get_engine

    ensure_analytics_tables_exist()
    engine = get_engine()
    with Session(engine) as db:
        row = db.execute(select(RequestTrace).where(RequestTrace.trace_id == str(trace_id))).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="Trace not found")
        return {
            "trace_id": row.trace_id,
            "ts": int(row.ts or 0),
            "tenant_id": row.tenant_id,
            "branch_id": row.branch_id,
            "channel": row.channel,
            "session_id": row.session_id,
            "user_id": row.user_id,
            "route": row.route,
            "status": row.status,
            "latency_ms": row.latency_ms,
            "sources": list(row.sources or []) if isinstance(row.sources, list) else [],
            "tool_metadata": dict(row.tool_metadata or {}) if isinstance(row.tool_metadata, dict) else {},
            "question": (row.question or "")[:20000],
            "answer": (row.answer or "")[:20000],
            "error": row.error,
        }


@router.get("/owner/api/handoffs")
def owner_handoffs(
    tenant_id: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    _owner: Dict[str, Any] = Depends(require_owner),
) -> Dict[str, Any]:
    rows = list_handoffs(
        tenant_id=tenant_id,
        since_ts=_parse_date_like(since),
        until_ts=_parse_date_like(until),
        status=status,
        limit=limit,
        offset=offset,
    )
    return {"rows": rows}
