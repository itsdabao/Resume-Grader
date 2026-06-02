import sys
import logging
import argparse
import traceback
from pathlib import Path
import os
import subprocess

# Đảm bảo import được gói src/* khi chạy từ thư mục scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app.core.config as cfg
from app.core.bootstrap import bootstrap_runtime
from app.services.rag_service import build_index, rag_query
from app.services.memory.store import get_or_create_session, save_session


def _ensure_agent_env() -> None:
    """
    Re-exec in conda env `agent` so dependencies (llama-index, sqlalchemy, psycopg2) are present.
    """
    if os.getenv("QUERY_NO_REEXEC"):
        return
    try:
        import llama_index  # type: ignore  # noqa: F401
        import sqlalchemy  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    cmd = ["conda", "run", "-n", "agent", "python", "-X", "utf8", str(Path(__file__).resolve()), *sys.argv[1:]]
    env = dict(os.environ)
    env["QUERY_NO_REEXEC"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        r = subprocess.run(cmd, env=env)
        raise SystemExit(r.returncode)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Cannot find `conda` to re-exec into env `agent`. "
            "Run inside the correct env:\n"
            "  conda run -n agent python scripts/query.py"
        ) from e


def init_logging() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )


def main() -> None:
    _ensure_agent_env()
    init_logging()
    parser = argparse.ArgumentParser(description="Hybrid RAG (In-Context RALM)")
    parser.add_argument("--debug", action="store_true", help="Bật debug chi tiết")
    parser.add_argument(
        "--mode",
        choices=["vector", "hybrid", "hybrid_rerank"],
        default=None,
        help="Chế độ: vector | hybrid | hybrid_rerank",
    )
    parser.add_argument("--tenant", default=None, help="Tenant ID để lọc dữ liệu truy hồi")
    parser.add_argument("--branch", default=None, help="Branch ID (tuỳ chọn) để lọc theo chi nhánh")
    parser.add_argument(
        "--session",
        default=None,
        help="Session id cho memory bền vững (CLI). Dùng 'tenant_id:session_id' hoặc cung cấp --tenant + --session.",
    )
    args = parser.parse_args()

    # Ghi đè cấu hình theo cờ tối giản
    if args.debug:
        cfg.DEBUG_VERBOSE = True
        cfg.DEBUG_SHOW_PROMPT = True
    if args.mode:
        if args.mode == "vector":
            cfg.USE_BM25 = False
            cfg.RERANK_USE_COSINE = False
        elif args.mode == "hybrid":
            cfg.USE_BM25 = True
            cfg.RERANK_USE_COSINE = False
        else:  # hybrid_rerank
            cfg.USE_BM25 = True
            cfg.RERANK_USE_COSINE = True

    print("Đang khởi tạo. Vui lòng chờ...")

    try:
        bootstrap_runtime()
        build_index()
        print('--- Sẵn sàng. Nhập câu hỏi (gõ "exit" để thoát, gõ "/reset" để xoá lịch sử) ---')

        if args.tenant:
            print(f"Tenant: {args.tenant}")
        if args.branch:
            print(f"Branch: {args.branch}")
        history: list[dict] = []
        session_id = None
        if args.session:
            s = str(args.session).strip()
            if ":" in s:
                session_id = s
            elif args.tenant:
                session_id = f"{args.tenant}:{s}"
            else:
                session_id = s
        while True:
            user_query = input("\nBạn hỏi: ")
            if user_query.strip().lower() == "exit":
                print("Tạm biệt!")
                break
            if user_query.strip().lower() == "/reset":
                history.clear()
                print("Đã xoá lịch sử hội thoại.")
                if session_id and args.tenant:
                    try:
                        st = get_or_create_session(session_id=session_id, tenant_id=args.tenant)
                        st.rolling_summary = ""
                        st.recent_messages_buffer = []
                        st.entity_memory = {}
                        save_session(state=st)
                        print("Đã reset memory trong Postgres cho session này.")
                    except Exception as e:
                        print(f"Không reset được memory DB: {e}")
                continue

            result = rag_query(
                user_query,
                tenant_id=args.tenant,
                branch_id=args.branch,
                history=history,
                channel="cli",
                session_id=session_id,
            )

            print("\nAnswer:")
            print(result.get("answer", ""))
            sources = result.get("sources", [])
            if sources:
                print("\nSources:")
                for s in sources:
                    print(f" - {s}")

            # Lưu lịch sử hội thoại (stateful theo session CLI)
            if session_id is None:
                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": str(result.get("answer", ""))})
    except Exception as e:
        # Một số Exception có message rỗng -> in repr + traceback để debug dễ hơn.
        msg = str(e).strip()
        if msg:
            print(f"Đã xảy ra lỗi: {msg}")
        else:
            print(f"Đã xảy ra lỗi: {type(e).__name__}: {repr(e)}")
        if args.debug or not msg:
            traceback.print_exc()


if __name__ == "__main__":
    main()
