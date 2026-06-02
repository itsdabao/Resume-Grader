"""
Benchmark latency via the running server's HTTP /query endpoint.
Usage: python scripts/benchmark_latency.py
Requires: server running at localhost:8000
"""
import json
import time
import requests

BASE = "http://localhost:8000"

TEST_CASES = [
    ("Smalltalk (canned)",       "Xin chào",                                          "smalltalk"),
    ("Smalltalk (regex/joke)",   "Một thêm một là mấy?",                              "smalltalk"),
    ("Smalltalk (regex/joke2)",  "Bạn là ai?",                                        "smalltalk"),
    ("Out-of-domain",            "What is quantum computing?",                         "out_of_domain"),
    ("Course search",            "Trung tâm có khóa IELTS nào cho người mất gốc?",    "course_search"),
    ("Tuition calculator",       "Học phí khóa IELTS bao nhiêu?",                     "tuition_calculator"),
    ("Comparison",               "So sánh IELTS với TOEIC khác nhau thế nào?",        "comparison"),
    ("Create ticket",            "Cho tôi gặp tư vấn viên, SĐT 0901234567",          "create_ticket"),
]

TENANT = "brightpathacademy"
API_KEY = "0LJCmxYNAgoHnSo9Mv8pKjTZgtHaVeznvECfOwp2wD0"
HEADERS = {"Content-Type": "application/json", "x-api-key": API_KEY}


def main():
    print("=" * 72)
    print("  AI Agent — Latency Benchmark (via HTTP /query)")
    print("=" * 72)
    print()

    # Wait for server + models to be ready (poll /health for up to 5 min)
    print("⏳ Waiting for server and models to load...")
    for attempt in range(60):  # 60 x 5s = 5 minutes
        try:
            r = requests.get(f"{BASE}/health", timeout=5)
            data = r.json()
            if data.get("models_ready") == "true":
                print(f"✅ Server OK — models loaded ({(attempt+1)*5}s)\n")
                break
            else:
                status = data.get("status", "?")
                print(f"  ⏳ Server up but models loading... ({(attempt+1)*5}s) [status={status}]")
        except Exception:
            print(f"  ⏳ Waiting for server... ({(attempt+1)*5}s)")
        time.sleep(5)
    else:
        print(f"❌ Server/models not ready after 5 min")
        return

    results = []
    for label, question, expected in TEST_CASES:
        print(f"▶ {label}")
        print(f"  Q: {question}")

        body = {"question": question, "tenant_id": TENANT}
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{BASE}/query", json=body, headers=HEADERS, timeout=60)
            elapsed = (time.perf_counter() - t0) * 1000
            data = resp.json()
            answer = str(data.get("answer", ""))[:100]
            route = data.get("route", "?")
            sources = len(data.get("sources", []))
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            answer = f"ERROR: {e}"
            route = "error"
            sources = 0

        match = expected in str(route)
        icon = "✅" if match else "⚠️"
        color_ms = f"{elapsed:.0f}ms"
        if elapsed < 1000:
            color_ms = f"{elapsed:.0f}ms  🟢"
        elif elapsed < 3000:
            color_ms = f"{elapsed:.0f}ms  🟡"
        else:
            color_ms = f"{elapsed:.0f}ms  🔴"

        print(f"  {icon} Route: {route:<22} Time: {color_ms}")
        print(f"  A: {answer}...")
        print()

        results.append({
            "label": label,
            "question": question,
            "expected_route": expected,
            "actual_route": route,
            "time_ms": round(elapsed, 1),
            "answer_preview": answer,
            "sources": sources,
            "match": match,
        })

    # Summary
    print("=" * 72)
    print(f"  {'Test':<25} {'Route':<22} {'Time':>10}  {'Status'}")
    print("-" * 72)
    total = 0
    for r in results:
        status = "✅" if r["match"] else "⚠️ MISMATCH"
        ms = r["time_ms"]
        total += ms
        tag = "🟢" if ms < 1000 else ("🟡" if ms < 3000 else "🔴")
        print(f"  {r['label']:<25} {str(r['actual_route']):<22} {ms:>7.0f}ms {tag}  {status}")

    avg = total / len(results)
    print("-" * 72)
    print(f"  {'AVERAGE':<25} {'':22} {avg:>7.0f}ms")
    print(f"  {'TOTAL':<25} {'':22} {total:>7.0f}ms")
    print()
    under_3s = sum(1 for r in results if r["time_ms"] < 3000)
    print(f"  Target <3s: {under_3s}/{len(results)} passed")
    print("=" * 72)

    # Save
    out_path = "scripts/benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "avg_ms": round(avg, 1)}, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Saved to {out_path}")


if __name__ == "__main__":
    main()
