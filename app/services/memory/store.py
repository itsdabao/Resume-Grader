from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Column, Float, Index, Integer, String, Text, TIMESTAMP, create_engine, event, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session, declarative_base

from app.core.config import CHAT_SESSIONS_TABLE, DATABASE_URL


Base = declarative_base()


class ChatSession(Base):
    __tablename__ = CHAT_SESSIONS_TABLE

    id = Column(String(50), primary_key=True)
    tenant_id = Column(String(50), nullable=False)
    entity_memory = Column(JSONB().with_variant(JSON(), "sqlite"), nullable=False, default=dict)
    rolling_summary = Column(Text, nullable=True)
    recent_messages_buffer = Column(JSONB().with_variant(JSON(), "sqlite"), nullable=False, default=list)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("idx_chat_sessions_tenant", "tenant_id"),)


class CustomerProfile(Base):
    __tablename__ = "customer_profiles"

    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(50), nullable=False)
    user_id = Column(String(120), nullable=False)
    profile_json = Column(JSONB().with_variant(JSON(), "sqlite"), nullable=False, default=dict)
    consent_status = Column(String(20), nullable=False, default="unknown")  # unknown|granted|revoked
    ttl_until_ts = Column(Integer, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_customer_profiles_tenant_user", "tenant_id", "user_id", unique=True),
        Index("idx_customer_profiles_tenant", "tenant_id"),
    )


class MemoryFact(Base):
    __tablename__ = "memory_facts"

    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(50), nullable=False)
    user_id = Column(String(120), nullable=False)
    session_id = Column(String(120), nullable=True)
    fact_key = Column(String(80), nullable=False)
    fact_value_text = Column(Text, nullable=True)
    fact_value_json = Column(JSONB().with_variant(JSON(), "sqlite"), nullable=False, default=dict)
    confidence = Column(Float, nullable=True)
    source_route = Column(String(50), nullable=True)
    source_trace_id = Column(String(64), nullable=True)
    last_seen_ts = Column(Integer, nullable=False)
    expires_at_ts = Column(Integer, nullable=True)
    is_deleted = Column(Integer, nullable=False, default=0)  # 0=active, 1=deleted
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_memory_facts_tenant_user", "tenant_id", "user_id"),
        Index("idx_memory_facts_lookup", "tenant_id", "user_id", "fact_key"),
        Index("idx_memory_facts_expiry", "expires_at_ts"),
    )


class MemoryEvent(Base):
    __tablename__ = "memory_events"

    id = Column(String(64), primary_key=True)
    ts = Column(Integer, nullable=False)
    tenant_id = Column(String(50), nullable=False)
    user_id = Column(String(120), nullable=False)
    session_id = Column(String(120), nullable=True)
    event_type = Column(String(60), nullable=False)
    content_text = Column(Text, nullable=True)
    payload = Column(JSONB().with_variant(JSON(), "sqlite"), nullable=False, default=dict)
    source_route = Column(String(50), nullable=True)
    source_trace_id = Column(String(64), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_memory_events_tenant_user_ts", "tenant_id", "user_id", "ts"),
        Index("idx_memory_events_type", "event_type"),
    )


_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set. Please export DATABASE_URL or put it in .env.")
    _ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
    if DATABASE_URL.startswith("sqlite"):
        # Local Windows environments in this project can hit sqlite journal I/O issues.
        # Force lightweight journaling mode per connection for local dev reliability.
        @event.listens_for(_ENGINE, "connect")
        def _sqlite_local_pragma(dbapi_connection, connection_record):  # type: ignore[no-redef]
            cur = dbapi_connection.cursor()
            try:
                cur.execute("PRAGMA journal_mode=MEMORY;")
                cur.execute("PRAGMA synchronous=NORMAL;")
            finally:
                cur.close()
    return _ENGINE


def ensure_tables_exist() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine, checkfirst=True)


@dataclass
class SessionState:
    id: str
    tenant_id: str
    entity_memory: Dict[str, Any]
    rolling_summary: str
    recent_messages_buffer: List[Dict[str, Any]]
    updated_at: Optional[float] = None


def _now_ts() -> int:
    return int(time.time())


def _new_id() -> str:
    return uuid.uuid4().hex


def get_or_create_session(*, session_id: str, tenant_id: str) -> SessionState:
    ensure_tables_exist()
    engine = get_engine()
    with Session(engine) as db:
        row = db.get(ChatSession, session_id)
        if row is None:
            row = ChatSession(
                id=session_id,
                tenant_id=tenant_id,
                entity_memory={},
                rolling_summary="",
                recent_messages_buffer=[],
            )
            db.add(row)
            db.commit()
            db.refresh(row)
        # Fail-closed: if tenant_id mismatches, do not leak other tenant's session.
        if str(row.tenant_id) != str(tenant_id):
            raise RuntimeError("chat_sessions tenant_id mismatch for session_id (refuse to load).")
        return SessionState(
            id=str(row.id),
            tenant_id=str(row.tenant_id),
            entity_memory=(row.entity_memory or {}) if isinstance(row.entity_memory, dict) else {},
            rolling_summary=str(row.rolling_summary or ""),
            recent_messages_buffer=list(row.recent_messages_buffer or []) if isinstance(row.recent_messages_buffer, list) else [],
            updated_at=None,
        )


def save_session(*, state: SessionState) -> None:
    ensure_tables_exist()
    engine = get_engine()
    with Session(engine) as db:
        row = db.get(ChatSession, state.id)
        if row is None:
            row = ChatSession(
                id=state.id,
                tenant_id=state.tenant_id,
                entity_memory=state.entity_memory or {},
                rolling_summary=state.rolling_summary or "",
                recent_messages_buffer=state.recent_messages_buffer or [],
            )
            db.add(row)
        else:
            if str(row.tenant_id) != str(state.tenant_id):
                raise RuntimeError("chat_sessions tenant_id mismatch for session_id (refuse to update).")
            row.entity_memory = state.entity_memory or {}
            row.rolling_summary = state.rolling_summary or ""
            row.recent_messages_buffer = state.recent_messages_buffer or []
        db.commit()


def append_messages(
    *,
    state: SessionState,
    messages: List[Dict[str, Any]],
    max_messages: int,
) -> SessionState:
    buf = list(state.recent_messages_buffer or [])
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = str(m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        buf.append({"role": role, "content": content, "ts": int(m.get("ts") or _now_ts())})
    if max_messages > 0 and len(buf) > max_messages:
        buf = buf[-max_messages:]
    state.recent_messages_buffer = buf
    return state


def merge_entity_memory(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_entity_memory(out.get(k) or {}, v)
        else:
            out[k] = v
    return out


def upsert_customer_profile(
    *,
    tenant_id: str,
    user_id: str,
    profile_patch: Optional[Dict[str, Any]] = None,
    consent_status: Optional[str] = None,
    ttl_until_ts: Optional[int] = None,
) -> Dict[str, Any]:
    ensure_tables_exist()
    engine = get_engine()
    with Session(engine) as db:
        row = db.execute(
            select(CustomerProfile).where(
                CustomerProfile.tenant_id == str(tenant_id),
                CustomerProfile.user_id == str(user_id),
            )
        ).scalar_one_or_none()
        if row is None:
            row = CustomerProfile(
                id=_new_id(),
                tenant_id=str(tenant_id),
                user_id=str(user_id),
                profile_json={},
                consent_status=str(consent_status or "unknown"),
                ttl_until_ts=(int(ttl_until_ts) if ttl_until_ts is not None else None),
            )
            db.add(row)

        row.profile_json = merge_entity_memory(
            row.profile_json if isinstance(row.profile_json, dict) else {},
            profile_patch or {},
        )
        if consent_status is not None:
            row.consent_status = str(consent_status)
        if ttl_until_ts is not None:
            row.ttl_until_ts = int(ttl_until_ts)
        db.commit()
        db.refresh(row)

        return {
            "id": str(row.id),
            "tenant_id": str(row.tenant_id),
            "user_id": str(row.user_id),
            "profile_json": dict(row.profile_json or {}) if isinstance(row.profile_json, dict) else {},
            "consent_status": str(row.consent_status or "unknown"),
            "ttl_until_ts": int(row.ttl_until_ts) if row.ttl_until_ts is not None else None,
        }


def get_customer_profile(*, tenant_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    ensure_tables_exist()
    engine = get_engine()
    with Session(engine) as db:
        row = db.execute(
            select(CustomerProfile).where(
                CustomerProfile.tenant_id == str(tenant_id),
                CustomerProfile.user_id == str(user_id),
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        now_ts = _now_ts()
        if row.ttl_until_ts is not None and int(row.ttl_until_ts) <= now_ts:
            return None
        return {
            "id": str(row.id),
            "tenant_id": str(row.tenant_id),
            "user_id": str(row.user_id),
            "profile_json": dict(row.profile_json or {}) if isinstance(row.profile_json, dict) else {},
            "consent_status": str(row.consent_status or "unknown"),
            "ttl_until_ts": int(row.ttl_until_ts) if row.ttl_until_ts is not None else None,
        }


def add_memory_fact(
    *,
    tenant_id: str,
    user_id: str,
    fact_key: str,
    fact_value_text: Optional[str] = None,
    fact_value_json: Optional[Dict[str, Any]] = None,
    confidence: Optional[float] = None,
    session_id: Optional[str] = None,
    source_route: Optional[str] = None,
    source_trace_id: Optional[str] = None,
    expires_at_ts: Optional[int] = None,
) -> str:
    if not str(fact_key or "").strip():
        raise ValueError("fact_key is required")

    ensure_tables_exist()
    engine = get_engine()
    with Session(engine) as db:
        row = db.execute(
            select(MemoryFact).where(
                MemoryFact.tenant_id == str(tenant_id),
                MemoryFact.user_id == str(user_id),
                MemoryFact.fact_key == str(fact_key),
                MemoryFact.fact_value_text == (str(fact_value_text) if fact_value_text else None),
                MemoryFact.is_deleted == 0,
            )
        ).scalar_one_or_none()
        if row is None:
            row = MemoryFact(
                id=_new_id(),
                tenant_id=str(tenant_id),
                user_id=str(user_id),
                session_id=(str(session_id) if session_id else None),
                fact_key=str(fact_key),
                fact_value_text=(str(fact_value_text) if fact_value_text else None),
                fact_value_json=(fact_value_json or {}) if isinstance(fact_value_json or {}, dict) else {},
                confidence=(float(confidence) if confidence is not None else None),
                source_route=(str(source_route) if source_route else None),
                source_trace_id=(str(source_trace_id) if source_trace_id else None),
                last_seen_ts=_now_ts(),
                expires_at_ts=(int(expires_at_ts) if expires_at_ts is not None else None),
                is_deleted=0,
            )
            db.add(row)
        else:
            row.fact_value_json = merge_entity_memory(
                row.fact_value_json if isinstance(row.fact_value_json, dict) else {},
                fact_value_json or {},
            )
            if confidence is not None:
                row.confidence = float(confidence)
            if session_id:
                row.session_id = str(session_id)
            if source_route:
                row.source_route = str(source_route)
            if source_trace_id:
                row.source_trace_id = str(source_trace_id)
            if expires_at_ts is not None:
                row.expires_at_ts = int(expires_at_ts)
            row.last_seen_ts = _now_ts()
            row.is_deleted = 0
        db.commit()
        return str(row.id)


def list_memory_facts(
    *,
    tenant_id: str,
    user_id: str,
    limit: int = 50,
    include_expired: bool = False,
) -> List[Dict[str, Any]]:
    ensure_tables_exist()
    engine = get_engine()
    stmt = select(MemoryFact).where(
        MemoryFact.tenant_id == str(tenant_id),
        MemoryFact.user_id == str(user_id),
        MemoryFact.is_deleted == 0,
    )
    if not include_expired:
        now_ts = _now_ts()
        stmt = stmt.where((MemoryFact.expires_at_ts.is_(None)) | (MemoryFact.expires_at_ts > int(now_ts)))
    stmt = stmt.order_by(MemoryFact.last_seen_ts.desc()).limit(max(1, min(int(limit), 200)))

    out: List[Dict[str, Any]] = []
    with Session(engine) as db:
        for row in db.execute(stmt).scalars().all():
            out.append(
                {
                    "id": str(row.id),
                    "tenant_id": str(row.tenant_id),
                    "user_id": str(row.user_id),
                    "session_id": str(row.session_id) if row.session_id else None,
                    "fact_key": str(row.fact_key),
                    "fact_value_text": str(row.fact_value_text) if row.fact_value_text else None,
                    "fact_value_json": dict(row.fact_value_json or {}) if isinstance(row.fact_value_json, dict) else {},
                    "confidence": float(row.confidence) if row.confidence is not None else None,
                    "source_route": str(row.source_route) if row.source_route else None,
                    "source_trace_id": str(row.source_trace_id) if row.source_trace_id else None,
                    "last_seen_ts": int(row.last_seen_ts or 0),
                    "expires_at_ts": int(row.expires_at_ts) if row.expires_at_ts is not None else None,
                }
            )
    return out


def soft_delete_memory_facts(
    *,
    tenant_id: str,
    user_id: str,
    fact_key: Optional[str] = None,
    fact_value_text: Optional[str] = None,
) -> int:
    ensure_tables_exist()
    engine = get_engine()
    with Session(engine) as db:
        stmt = select(MemoryFact).where(
            MemoryFact.tenant_id == str(tenant_id),
            MemoryFact.user_id == str(user_id),
            MemoryFact.is_deleted == 0,
        )
        if fact_key:
            stmt = stmt.where(MemoryFact.fact_key == str(fact_key))
        if fact_value_text is not None:
            stmt = stmt.where(MemoryFact.fact_value_text == str(fact_value_text))

        rows = db.execute(stmt).scalars().all()
        n = 0
        for row in rows:
            row.is_deleted = 1
            row.last_seen_ts = _now_ts()
            n += 1
        if n > 0:
            db.commit()
        return int(n)


def add_memory_event(
    *,
    tenant_id: str,
    user_id: str,
    event_type: str,
    content_text: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    source_route: Optional[str] = None,
    source_trace_id: Optional[str] = None,
    ts: Optional[int] = None,
) -> str:
    if not str(event_type or "").strip():
        raise ValueError("event_type is required")

    ensure_tables_exist()
    engine = get_engine()
    with Session(engine) as db:
        row = MemoryEvent(
            id=_new_id(),
            ts=int(ts or _now_ts()),
            tenant_id=str(tenant_id),
            user_id=str(user_id),
            session_id=(str(session_id) if session_id else None),
            event_type=str(event_type),
            content_text=(str(content_text) if content_text else None),
            payload=(payload or {}) if isinstance(payload or {}, dict) else {},
            source_route=(str(source_route) if source_route else None),
            source_trace_id=(str(source_trace_id) if source_trace_id else None),
        )
        db.add(row)
        db.commit()
        return str(row.id)


def list_memory_events(
    *,
    tenant_id: str,
    user_id: str,
    event_type: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    ensure_tables_exist()
    engine = get_engine()
    stmt = select(MemoryEvent).where(
        MemoryEvent.tenant_id == str(tenant_id),
        MemoryEvent.user_id == str(user_id),
    )
    if event_type:
        stmt = stmt.where(MemoryEvent.event_type == str(event_type))
    stmt = stmt.order_by(MemoryEvent.ts.desc()).limit(max(1, min(int(limit), 200)))

    out: List[Dict[str, Any]] = []
    with Session(engine) as db:
        for row in db.execute(stmt).scalars().all():
            out.append(
                {
                    "id": str(row.id),
                    "ts": int(row.ts or 0),
                    "tenant_id": str(row.tenant_id),
                    "user_id": str(row.user_id),
                    "session_id": str(row.session_id) if row.session_id else None,
                    "event_type": str(row.event_type),
                    "content_text": str(row.content_text) if row.content_text else None,
                    "payload": dict(row.payload or {}) if isinstance(row.payload, dict) else {},
                    "source_route": str(row.source_route) if row.source_route else None,
                    "source_trace_id": str(row.source_trace_id) if row.source_trace_id else None,
                }
            )
    return out
