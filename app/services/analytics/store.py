from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import JSON, Column, Float, Integer, String, Text, TIMESTAMP, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session, declarative_base

from app.services.memory.store import get_engine


Base = declarative_base()


class RequestTrace(Base):
    __tablename__ = "request_traces"

    trace_id = Column(String(64), primary_key=True)
    ts = Column(Integer, nullable=False)  # epoch seconds

    tenant_id = Column(String(50), nullable=True)
    branch_id = Column(String(50), nullable=True)
    channel = Column(String(30), nullable=False, default="api")

    session_id = Column(String(120), nullable=True)
    user_id = Column(String(120), nullable=True)

    route = Column(String(50), nullable=True)
    status = Column(String(20), nullable=False, default="SUCCESS")  # SUCCESS|ERROR
    latency_ms = Column(Float, nullable=True)

    question = Column(Text, nullable=True)
    answer = Column(Text, nullable=True)
    sources = Column(JSONB().with_variant(JSON(), "sqlite"), nullable=False, default=list)
    tool_metadata = Column(JSONB().with_variant(JSON(), "sqlite"), nullable=False, default=dict)
    error = Column(Text, nullable=True)


class Feedback(Base):
    __tablename__ = "user_feedback"

    id = Column(String(64), primary_key=True)
    ts = Column(Integer, nullable=False)
    trace_id = Column(String(64), nullable=False, index=True)
    tenant_id = Column(String(50), nullable=True)
    rating = Column(Integer, nullable=False)  # 1=up, -1=down
    comment = Column(Text, nullable=True)


class HandoffTicket(Base):
    __tablename__ = "handoff_tickets"

    id = Column(String(64), primary_key=True)
    ts = Column(Integer, nullable=False)
    tenant_id = Column(String(50), nullable=True)
    branch_id = Column(String(50), nullable=True)
    user_id = Column(String(120), nullable=True)
    phone = Column(String(40), nullable=True)
    message = Column(Text, nullable=True)
    status = Column(String(30), nullable=False, default="new")  # new|contacted|closed
    meta = Column(JSONB().with_variant(JSON(), "sqlite"), nullable=False, default=dict)


def ensure_analytics_tables_exist() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine, checkfirst=True)


def _now_ts() -> int:
    return int(time.time())


def new_trace_id() -> str:
    return uuid.uuid4().hex


def insert_trace(
    *,
    trace_id: str,
    tenant_id: Optional[str],
    branch_id: Optional[str],
    channel: str,
    session_id: Optional[str],
    user_id: Optional[str],
    question: str,
    answer: str,
    sources: List[str],
    route: Optional[str],
    status: str,
    latency_ms: Optional[float],
    tool_metadata: Dict[str, Any] | None = None,
    error: Optional[str] = None,
) -> None:
    ensure_analytics_tables_exist()
    engine = get_engine()
    with Session(engine) as db:
        row = RequestTrace(
            trace_id=str(trace_id),
            ts=_now_ts(),
            tenant_id=(str(tenant_id) if tenant_id else None),
            branch_id=(str(branch_id) if branch_id else None),
            channel=str(channel or "api"),
            session_id=(str(session_id) if session_id else None),
            user_id=(str(user_id) if user_id else None),
            route=(str(route) if route else None),
            status=str(status or "SUCCESS"),
            latency_ms=float(latency_ms) if latency_ms is not None else None,
            question=str(question or ""),
            answer=str(answer or ""),
            sources=[str(s) for s in (sources or [])],
            tool_metadata=(tool_metadata or {}) if isinstance(tool_metadata or {}, dict) else {},
            error=str(error) if error else None,
        )
        db.add(row)
        db.commit()


def insert_feedback(*, trace_id: str, tenant_id: Optional[str], rating: int, comment: Optional[str] = None) -> str:
    ensure_analytics_tables_exist()
    engine = get_engine()
    fb_id = uuid.uuid4().hex
    with Session(engine) as db:
        row = Feedback(
            id=fb_id,
            ts=_now_ts(),
            trace_id=str(trace_id),
            tenant_id=(str(tenant_id) if tenant_id else None),
            rating=int(rating),
            comment=(str(comment) if comment else None),
        )
        db.add(row)
        db.commit()
    return fb_id


def insert_handoff_ticket(
    *,
    tenant_id: Optional[str],
    branch_id: Optional[str],
    user_id: Optional[str],
    phone: Optional[str],
    message: str,
    status: str = "new",
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    ensure_analytics_tables_exist()
    engine = get_engine()
    tid = uuid.uuid4().hex
    with Session(engine) as db:
        row = HandoffTicket(
            id=tid,
            ts=_now_ts(),
            tenant_id=(str(tenant_id) if tenant_id else None),
            branch_id=(str(branch_id) if branch_id else None),
            user_id=(str(user_id) if user_id else None),
            phone=(str(phone) if phone else None),
            message=str(message or ""),
            status=str(status or "new"),
            meta=(meta or {}) if isinstance(meta or {}, dict) else {},
        )
        db.add(row)
        db.commit()
    return tid


def list_tenants() -> List[str]:
    ensure_analytics_tables_exist()
    engine = get_engine()
    with Session(engine) as db:
        rows = db.execute(select(RequestTrace.tenant_id).where(RequestTrace.tenant_id.is_not(None))).all()
    uniq = sorted({str(r[0]) for r in rows if r and r[0]})
    return uniq


def list_traces(
    *,
    tenant_id: Optional[str],
    since_ts: Optional[int],
    until_ts: Optional[int],
    route: Optional[str],
    status: Optional[str],
    q: Optional[str],
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    ensure_analytics_tables_exist()
    engine = get_engine()
    stmt = select(RequestTrace).order_by(RequestTrace.ts.desc())
    if tenant_id:
        stmt = stmt.where(RequestTrace.tenant_id == str(tenant_id))
    if since_ts:
        stmt = stmt.where(RequestTrace.ts >= int(since_ts))
    if until_ts:
        stmt = stmt.where(RequestTrace.ts <= int(until_ts))
    if route:
        stmt = stmt.where(RequestTrace.route == str(route))
    if status:
        stmt = stmt.where(RequestTrace.status == str(status))
    if q:
        qq = f"%{q}%"
        stmt = stmt.where((RequestTrace.question.ilike(qq)) | (RequestTrace.answer.ilike(qq)))
    stmt = stmt.limit(max(1, min(int(limit), 500))).offset(max(0, int(offset)))

    out: List[Dict[str, Any]] = []
    with Session(engine) as db:
        for row in db.execute(stmt).scalars().all():
            out.append(
                {
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
                    "sources_count": len(row.sources or []) if isinstance(row.sources, list) else 0,
                    "question": (row.question or "")[:5000],
                    "answer": (row.answer or "")[:5000],
                    "error": row.error,
                }
            )
    return out


def list_handoffs(
    *,
    tenant_id: Optional[str],
    since_ts: Optional[int],
    until_ts: Optional[int],
    status: Optional[str],
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    ensure_analytics_tables_exist()
    engine = get_engine()
    stmt = select(HandoffTicket).order_by(HandoffTicket.ts.desc())
    if tenant_id:
        stmt = stmt.where(HandoffTicket.tenant_id == str(tenant_id))
    if since_ts:
        stmt = stmt.where(HandoffTicket.ts >= int(since_ts))
    if until_ts:
        stmt = stmt.where(HandoffTicket.ts <= int(until_ts))
    if status:
        stmt = stmt.where(HandoffTicket.status == str(status))
    stmt = stmt.limit(max(1, min(int(limit), 500))).offset(max(0, int(offset)))

    out: List[Dict[str, Any]] = []
    with Session(engine) as db:
        for row in db.execute(stmt).scalars().all():
            out.append(
                {
                    "id": row.id,
                    "ts": int(row.ts or 0),
                    "tenant_id": row.tenant_id,
                    "branch_id": row.branch_id,
                    "user_id": row.user_id,
                    "phone": row.phone,
                    "status": row.status,
                    "message": (row.message or "")[:5000],
                }
            )
    return out


def _percentile(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    x = sorted([float(v) for v in vals if v is not None])
    if not x:
        return None
    if p <= 0:
        return float(x[0])
    if p >= 1:
        return float(x[-1])
    k = (len(x) - 1) * p
    f = int(k)
    c = min(f + 1, len(x) - 1)
    if f == c:
        return float(x[f])
    d = k - f
    return float(x[f] * (1 - d) + x[c] * d)


def metrics(
    *,
    tenant_id: Optional[str],
    since_ts: Optional[int],
    until_ts: Optional[int],
) -> Dict[str, Any]:
    ensure_analytics_tables_exist()
    engine = get_engine()
    stmt = select(RequestTrace.latency_ms, RequestTrace.status, RequestTrace.route).where(RequestTrace.latency_ms.is_not(None))
    if tenant_id:
        stmt = stmt.where(RequestTrace.tenant_id == str(tenant_id))
    if since_ts:
        stmt = stmt.where(RequestTrace.ts >= int(since_ts))
    if until_ts:
        stmt = stmt.where(RequestTrace.ts <= int(until_ts))

    latencies: List[float] = []
    total_requests = 0
    error_requests = 0
    with Session(engine) as db:
        rows = db.execute(stmt).all()
        for latency_ms, status, _route in rows:
            total_requests += 1
            if str(status or "").upper() == "ERROR":
                error_requests += 1
            if latency_ms is not None:
                latencies.append(float(latency_ms))

        # Feedback
        fb_stmt = select(Feedback.rating).join(RequestTrace, Feedback.trace_id == RequestTrace.trace_id)
        if tenant_id:
            fb_stmt = fb_stmt.where(RequestTrace.tenant_id == str(tenant_id))
        if since_ts:
            fb_stmt = fb_stmt.where(RequestTrace.ts >= int(since_ts))
        if until_ts:
            fb_stmt = fb_stmt.where(RequestTrace.ts <= int(until_ts))
        fb_rows = [int(r[0]) for r in db.execute(fb_stmt).all() if r and isinstance(r[0], (int, float))]
        up = sum(1 for r in fb_rows if int(r) > 0)
        down = sum(1 for r in fb_rows if int(r) < 0)

        # Handoff rate = tickets / total_requests (same time window)
        t_stmt = select(func.count(HandoffTicket.id))
        if tenant_id:
            t_stmt = t_stmt.where(HandoffTicket.tenant_id == str(tenant_id))
        if since_ts:
            t_stmt = t_stmt.where(HandoffTicket.ts >= int(since_ts))
        if until_ts:
            t_stmt = t_stmt.where(HandoffTicket.ts <= int(until_ts))
        handoffs = int(db.execute(t_stmt).scalar() or 0)

    avg = (sum(latencies) / len(latencies)) if latencies else None
    p50 = _percentile(latencies, 0.5)
    p95 = _percentile(latencies, 0.95)
    satisfaction = (up / (up + down)) if (up + down) > 0 else None
    handoff_rate = (handoffs / total_requests) if total_requests > 0 else None

    return {
        "tenant_id": tenant_id,
        "total_requests": total_requests,
        "error_requests": error_requests,
        "avg_time_ms": round(avg, 1) if isinstance(avg, (int, float)) else None,
        "p50_ms": round(p50, 1) if isinstance(p50, (int, float)) else None,
        "p95_ms": round(p95, 1) if isinstance(p95, (int, float)) else None,
        "satisfaction_rate": round(float(satisfaction), 3) if isinstance(satisfaction, (int, float)) else None,
        "feedback_total": int(up + down),
        "handoff_count": int(handoffs),
        "handoff_rate": round(float(handoff_rate), 3) if isinstance(handoff_rate, (int, float)) else None,
    }

