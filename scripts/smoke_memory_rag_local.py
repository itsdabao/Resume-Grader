from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _set_local_env(db_url: str) -> None:
    # Local-first defaults for the smoke check.
    os.environ["DATABASE_URL"] = db_url
    os.environ.setdefault("MEMORY_ENABLED", "1")
    os.environ.setdefault("RAG_INIT_ON_STARTUP", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


def _run_memory_core_check() -> Dict[str, Any]:
    """
    Check 1: memory_rag_query persists and reloads chat history from SQLite.
    Uses a stubbed `agentic_query` so this test is deterministic and fast.
    """
    from app.services.memory import service as memory_service
    from app.services.memory import store as memory_store

    class _DummyIndex:
        pass

    seen_histories: List[List[Dict[str, str]]] = []
    original_agentic = memory_service.agentic_query

    def _fake_agentic_query(
        question: str,
        *,
        index: Any,
        tenant_id: str | None = None,
        branch_id: str | None = None,
        history: List[Dict[str, str]] | None = None,
        user_id: str | None = None,
    ) -> Dict[str, Any]:
        h = list(history or [])
        seen_histories.append(h)
        return {
            "answer": f"stub_answer(history_len={len(h)}): {question}",
            "sources": ["stub_source"],
            "route": "stub_agentic",
            "tool_metadata": {"history_len": len(h)},
        }

    memory_service.agentic_query = _fake_agentic_query

    tenant_id = "tenant_smoke"
    user_id = "user_smoke"
    sid = f"{tenant_id}:web:{user_id}:{int(time.time())}"

    try:
        r1 = memory_service.memory_rag_query(
            "Xin chao, toi ten la An.",
            index=_DummyIndex(),
            tenant_id=tenant_id,
            channel="web",
            user_id=user_id,
            session_id=sid,
        )
        r2 = memory_service.memory_rag_query(
            "Ban nhac lai toi vua noi gi?",
            index=_DummyIndex(),
            tenant_id=tenant_id,
            channel="web",
            user_id=user_id,
            session_id=sid,
        )
    finally:
        memory_service.agentic_query = original_agentic

    state = memory_store.get_or_create_session(session_id=sid, tenant_id=tenant_id)
    buffer_len = len(state.recent_messages_buffer or [])
    second_turn_history_len = len(seen_histories[1]) if len(seen_histories) >= 2 else -1

    ok = (
        len(seen_histories) == 2
        and len(seen_histories[0]) == 0
        and second_turn_history_len >= 2
        and buffer_len >= 4
        and str(r1.get("answer", "")).startswith("stub_answer")
        and str(r2.get("answer", "")).startswith("stub_answer")
    )
    return {
        "ok": bool(ok),
        "session_id": sid,
        "first_turn_history_len": len(seen_histories[0]) if seen_histories else -1,
        "second_turn_history_len": second_turn_history_len,
        "buffer_len_after_2_turns": buffer_len,
        "memory_preview": state.recent_messages_buffer[-4:] if buffer_len else [],
    }


def _run_real_rag_readiness() -> Dict[str, Any]:
    """
    Check 2: best-effort readiness for real path (Qdrant + LLM + memory_rag_query).
    This does not force success; it reports blockers clearly.
    """
    out: Dict[str, Any] = {
        "qdrant_ready": False,
        "embedding_ready": False,
        "llm_ready": False,
        "real_rag_ok": False,
    }

    try:
        from app.services.retrieval.vector_store import init_qdrant_collection

        _ = init_qdrant_collection()
        out["qdrant_ready"] = True
    except Exception as e:
        out["qdrant_error"] = str(e)

    try:
        from app.core.bootstrap import bootstrap_embeddings_only

        bootstrap_embeddings_only()
        out["embedding_ready"] = True
    except Exception as e:
        out["embedding_error"] = str(e)

    try:
        from llama_index.core import Settings
        from app.core.llama import init_llm_from_env

        init_llm_from_env()
        out["llm_ready"] = Settings.llm is not None
        if not out["llm_ready"]:
            out["llm_error"] = "Settings.llm is None (LLM disabled)."
    except Exception as e:
        out["llm_error"] = str(e)

    if out["qdrant_ready"] and out["embedding_ready"] and out["llm_ready"]:
        try:
            from app.services.rag_service import rag_query

            tenant_id = "tenant_smoke"
            user_id = "user_smoke_real"
            sid = f"{tenant_id}:web:{user_id}:{int(time.time())}"
            res = rag_query(
                question="Cho toi thong tin khoa hoc IELTS co ban.",
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=sid,
                channel="web",
            )
            out["real_rag_ok"] = True
            out["real_rag_preview"] = {
                "route": res.get("route"),
                "sources_count": len(res.get("sources", []) or []),
                "has_memory_meta": isinstance(res.get("memory"), dict),
            }
        except Exception as e:
            out["real_rag_error"] = str(e)
    return out


def _run_chat_api_wiring_check() -> Dict[str, Any]:
    """
    Check 3: ensure `/chat` route forwards tenant/session/user into rag_query.
    Uses dependency override (no Firebase needed in this smoke test).
    """
    from fastapi.testclient import TestClient

    import app.api.main as api_main

    captured: Dict[str, Any] = {}
    original_rag_query = api_main.rag_query

    def _fake_rag_query(
        question: str,
        *,
        tenant_id: str | None = None,
        branch_id: str | None = None,
        history: List[Dict[str, str]] | None = None,
        channel: str = "cli",
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> Dict[str, Any]:
        captured.update(
            {
                "question": question,
                "tenant_id": tenant_id,
                "branch_id": branch_id,
                "history_len": len(history or []),
                "channel": channel,
                "user_id": user_id,
                "session_id": session_id,
            }
        )
        return {"answer": "ok_chat", "sources": ["stub"], "route": "chat_stub", "tool_metadata": {}}

    api_main.rag_query = _fake_rag_query
    api_main.app.dependency_overrides[api_main.get_current_user] = lambda: {
        "uid": "u_api",
        "tenant_id": "tenant_api",
        "email": "u_api@example.com",
    }

    try:
        with TestClient(api_main.app) as client:
            resp = client.post(
                "/chat",
                json={"message": "Xin chao", "session_id": "tenant_api:web:u_api", "branch_id": "b1"},
            )
    finally:
        api_main.rag_query = original_rag_query
        api_main.app.dependency_overrides.pop(api_main.get_current_user, None)

    ok = (
        resp.status_code == 200
        and captured.get("tenant_id") == "tenant_api"
        and captured.get("user_id") == "u_api"
        and captured.get("session_id") == "tenant_api:web:u_api"
        and captured.get("channel") == "tenant_chat"
    )
    body: Dict[str, Any] = {}
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}

    return {
        "ok": bool(ok),
        "status_code": resp.status_code,
        "response": body,
        "captured_rag_args": captured,
    }


def _run_query_api_wiring_check() -> Dict[str, Any]:
    """
    Check 4: ensure `/query` can be used (with x-api-key) and forwards memory fields.
    Useful before Firebase integration.
    """
    from fastapi.testclient import TestClient

    import app.api.main as api_main

    captured: Dict[str, Any] = {}
    original_rag_query = api_main.rag_query

    def _fake_rag_query(
        question: str,
        *,
        tenant_id: str | None = None,
        branch_id: str | None = None,
        history: List[Dict[str, str]] | None = None,
        channel: str = "cli",
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> Dict[str, Any]:
        captured.update(
            {
                "question": question,
                "tenant_id": tenant_id,
                "branch_id": branch_id,
                "history_len": len(history or []),
                "channel": channel,
                "user_id": user_id,
                "session_id": session_id,
            }
        )
        return {"answer": "ok_query", "sources": ["stub"], "route": "query_stub", "tool_metadata": {}}

    api_main.rag_query = _fake_rag_query
    key = str(getattr(api_main, "PUBLIC_QUERY_API_KEY", "") or "")

    try:
        with TestClient(api_main.app) as client:
            headers = {"x-api-key": key} if key else {}
            resp = client.post(
                "/query",
                headers=headers,
                json={
                    "question": "Test memory query",
                    "tenant_id": "tenant_query",
                    "branch_id": "b2",
                    "session_id": "tenant_query:web:user_q",
                    "user_id": "user_q",
                    "history": [],
                },
            )
    finally:
        api_main.rag_query = original_rag_query

    body: Dict[str, Any] = {}
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}

    ok = (
        bool(key)
        and resp.status_code == 200
        and captured.get("tenant_id") == "tenant_query"
        and captured.get("session_id") == "tenant_query:web:user_q"
        and captured.get("user_id") == "user_q"
        and captured.get("channel") == "web"
    )
    return {
        "ok": bool(ok),
        "status_code": resp.status_code,
        "public_query_api_key_present": bool(key),
        "response": body,
        "captured_rag_args": captured,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test memory_rag_query with local SQLite.")
    parser.add_argument(
        "--db-url",
        default="sqlite:///./agent_local.db",
        help="SQLAlchemy DATABASE_URL for local smoke run (default: sqlite:///./agent_local.db)",
    )
    parser.add_argument(
        "--skip-real-rag",
        action="store_true",
        help="Skip Qdrant+LLM readiness check and only run SQLite memory core check.",
    )
    args = parser.parse_args()

    _set_local_env(args.db_url)

    report: Dict[str, Any] = {
        "database_url": args.db_url,
        "memory_core": _run_memory_core_check(),
        "chat_api_wiring": _run_chat_api_wiring_check(),
        "query_api_wiring": _run_query_api_wiring_check(),
    }
    if not args.skip_real_rag:
        report["real_rag_readiness"] = _run_real_rag_readiness()

    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
