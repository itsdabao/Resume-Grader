import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_agent_env() -> None:
    """
    Re-exec in conda env `agent` to ensure deps exist (llama-index, etc.).
    """
    if os.getenv("EVAL_NO_REEXEC"):
        return
    try:
        import llama_index  # type: ignore  # noqa: F401

        return
    except Exception:
        pass

    cmd = ["conda", "run", "-n", "agent", "python", str(Path(__file__).resolve())]
    env = dict(os.environ)
    env["EVAL_NO_REEXEC"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    print("Re-running with: " + " ".join(cmd), flush=True)
    try:
        r = subprocess.run(cmd, env=env)
        raise SystemExit(r.returncode)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Cannot find `conda` to re-exec into env `agent`. "
            "Run inside the correct env:\n"
            "  conda run -n agent python scripts/eval_discount_tool.py"
        ) from e


CASES_PATH = Path("data/.cache/eval_discount_cases.jsonl")
REPORT_PATH = Path("data/.cache/eval_discount_report.json")


DEFAULT_CASES: List[Dict[str, Any]] = [
    {"id": "pct_10tr_10", "query": "Học phí 10tr giảm 10% còn bao nhiêu?", "expected": {"final_vnd": 9000000}},
    {"id": "pct_9500k_10", "query": "9.500.000đ giảm 10% còn bao nhiêu?", "expected": {"final_vnd": 8550000}},
    {"id": "amt_10tr_500k", "query": "Học phí 10tr giảm 500k còn bao nhiêu?", "expected": {"final_vnd": 9500000}},
    {"id": "amt_9500k_1tr", "query": "9.500.000 VND giảm 1tr còn bao nhiêu?", "expected": {"final_vnd": 8500000}},
    {
        "id": "pct_5tr_10_plus_hoc_lieu_500k",
        "query": "hoc phi 5tr giam 10%, hoc lieu 500k thi tong la bao nhieu?",
        "expected": {"final_vnd": 5000000},
    },
    {"id": "invalid_pct", "query": "10tr giảm 120% còn bao nhiêu?", "expected": {"error": True}},
    {"id": "invalid_amt", "query": "10tr giảm 20tr còn bao nhiêu?", "expected": {"error": True}},
    {"id": "missing_discount", "query": "10tr sau giảm còn bao nhiêu?", "expected": {"needs_more_info": True}},
]


def _load_cases() -> List[Dict[str, Any]]:
    if CASES_PATH.exists():
        out: List[Dict[str, Any]] = []
        for ln in CASES_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("query"):
                out.append(obj)
        if out:
            return out
    CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CASES_PATH.write_text("\n".join([json.dumps(x, ensure_ascii=False) for x in DEFAULT_CASES]) + "\n", encoding="utf-8")
    return list(DEFAULT_CASES)


def main() -> None:
    _ensure_agent_env()

    from app.services.agentic.preprocess import preprocess_query
    from app.services.agentic.router import route_query
    from app.services.agentic.tools import tuition_calculator_tool

    cases = _load_cases()
    t0 = time.perf_counter()

    results: List[Dict[str, Any]] = []
    n_pass = 0
    n_fail = 0

    for c in cases:
        case_id = str(c.get("id") or "")
        q = str(c.get("query") or "").strip()
        exp = c.get("expected") if isinstance(c.get("expected"), dict) else {}
        if not q:
            continue

        # 1) Router decision
        p = preprocess_query(q)
        decision = route_query(p)

        # 2) Run tool in direct mode (no index needed when base+discount is in query)
        tool = tuition_calculator_tool(q, index=None)
        md = tool.metadata or {}
        computed = md.get("computed_final_vnd")

        ok_route = decision.route == "tuition_calculator"
        ok = True
        reason = []

        if exp.get("final_vnd") is not None:
            ok = (computed == int(exp["final_vnd"])) and ok_route
            if not ok_route:
                reason.append(f"route={decision.route}")
            if computed != int(exp["final_vnd"]):
                reason.append(f"computed={computed} expected={int(exp['final_vnd'])}")
        elif exp.get("error"):
            ok = ("không hợp lệ" in (tool.answer or "").lower()) and ok_route
            if not ok:
                reason.append("expected_error")
        elif exp.get("needs_more_info"):
            ok = ("cho em xin" in (tool.answer or "").lower()) and ok_route
            if not ok:
                reason.append("expected_needs_more_info")

        results.append(
            {
                "id": case_id,
                "query": q,
                "route": decision.route,
                "route_reason": decision.reason,
                "tool_answer": tool.answer,
                "tool_metadata": md,
                "pass": bool(ok),
                "fail_reason": "; ".join(reason),
            }
        )
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    report = {
        "cases_total": len(results),
        "pass": n_pass,
        "fail": n_fail,
        "pass_rate": (n_pass / max(1, len(results))),
        "elapsed_ms": round(elapsed_ms, 1),
        "cases_path": str(CASES_PATH),
        "results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: wrote report to {REPORT_PATH} (pass={n_pass} fail={n_fail})")
    if n_fail:
        print("Hint: open the report and inspect fail_reason/tool_answer/tool_metadata.")


if __name__ == "__main__":
    main()


