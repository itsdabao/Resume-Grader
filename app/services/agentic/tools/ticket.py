import json
import time
from pathlib import Path
from typing import Optional

from app.core.config import PROJECT_ROOT
from app.services.agentic.preprocess import extract_phone
from app.services.agentic.tools.base import ToolResult


def create_ticket_tool(
    question: str,
    *,
    tenant_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> ToolResult:
    """
    Minimal ticket/handoff:
    - Best-effort write to Postgres (handoff_tickets) for dashboard metrics.
    - Also append to local JSONL for backward compatibility.
    """
    phone = extract_phone(question or "")
    if not phone:
        return ToolResult(
            answer="Dạ được ạ. Anh/chị cho em xin **SĐT** và **khung giờ thuận tiện** để tư vấn viên liên hệ hỗ trợ chi tiết nhé.",
            sources=[],
            metadata={"route": "create_ticket", "ticket_created": False},
        )

    ticket_id = None
    try:
        from app.services.analytics.store import insert_handoff_ticket

        ticket_id = insert_handoff_ticket(
            tenant_id=tenant_id,
            branch_id=branch_id,
            user_id=user_id,
            phone=phone,
            message=(question or "").strip(),
            status="new",
            meta={"source": "create_ticket_tool"},
        )
    except Exception:
        ticket_id = None

    ticket = {
        "ts": int(time.time()),
        "id": ticket_id,
        "tenant_id": tenant_id,
        "branch_id": branch_id,
        "user_id": user_id,
        "phone": phone,
        "message": (question or "").strip(),
        "status": "new",
    }
    out_path = Path(PROJECT_ROOT) / "data" / ".cache" / "tickets.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8") if not out_path.exists() else None
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ticket, ensure_ascii=False) + "\n")

    return ToolResult(
        answer="Dạ em đã ghi nhận thông tin. Tư vấn viên sẽ liên hệ với anh/chị sớm nhất ạ.",
        sources=[],
        metadata={"route": "create_ticket", "ticket_created": True, "ticket_id": ticket_id},
    )
