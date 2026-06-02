import sys
import time
import json
import asyncio
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.rag_service import rag_query
from app.core.llama import init_llm_from_env
from llama_index.core.llms import ChatMessage, MessageRole

def evaluate_with_llm(llm, question, answer, context):
    """Sử dụng LLM-as-a-judge để chấm điểm định lượng (Quantitative Metrics)."""
    prompt = f"""
You are an expert evaluator grading a RAG system.
Please evaluate the following RAG system response based on the provided Question and Retrieved Context.

Question: {question}
Retrieved Context: {context}
System Answer: {answer}

Rate the following two metrics on a scale of 1 to 5 (where 5 is best).
Return ONLY a valid JSON object with the following format, nothing else:
{{
    "answer_correctness": <int 1-5>,
    "retrieval_relevance": <int 1-5>,
    "qualitative_reasoning": "<short explanation>"
}}
"""
    try:
        response = llm.chat([ChatMessage(role=MessageRole.USER, content=prompt)])
        # Clean up code blocks if present
        text = response.message.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result
    except Exception as e:
        print(f"Lỗi khi dùng LLM chấm điểm: {e}")
        return {
            "answer_correctness": 0,
            "retrieval_relevance": 0,
            "qualitative_reasoning": f"Failed to parse evaluation: {e}"
        }

def run_evaluation():
    print("Khởi tạo mô hình LLM để đánh giá...")
    llm = init_llm_from_env()

    test_cases = [
        "Hãy tìm top 3 ứng viên có kinh nghiệm tốt nhất về Java Backend trong mảng Ngân hàng.",
        "Ứng viên nào có kỹ năng về UI/UX và Figma tốt nhất?",
        "Danh sách những ứng viên từng làm vị trí 'Data Analyst' hoặc liên quan đến dữ liệu.",
    ]

    results = []
    total_latency = 0

    print("\nBẮT ĐẦU ĐÁNH GIÁ (RAG EVALUATION)\n" + "="*50)

    for i, question in enumerate(test_cases, 1):
        print(f"\n[Test {i}/3] Đang xử lý: '{question}'")
        
        # Đo thời gian phản hồi (Latency)
        start_time = time.perf_counter()
        rag_response = rag_query(question=question, channel="evaluation")
        latency = (time.perf_counter() - start_time) * 1000  # ms
        total_latency += latency
        
        answer = rag_response.get("answer", "")
        sources = rag_response.get("sources", [])
        
        # Giả lập context ghép từ các sources
        context = " ".join([str(s) for s in sources[:3]]) # Lấy 3 source đầu tiên
        
        # Dùng LLM chấm điểm
        eval_metrics = evaluate_with_llm(llm, question, answer, context)
        
        test_result = {
            "question": question,
            "latency_ms": round(latency, 2),
            "answer": answer,
            "metrics": eval_metrics,
            "num_sources_retrieved": len(sources)
        }
        results.append(test_result)
        
        print(f"  -> Latency: {latency:.2f} ms")
        print(f"  -> Answer Correctness: {eval_metrics.get('answer_correctness')}/5")
        print(f"  -> Retrieval Relevance: {eval_metrics.get('retrieval_relevance')}/5")

    # Tạo file Markdown Report
    report_path = REPO_ROOT / "EVALUATION_REPORT.md"
    
    avg_latency = total_latency / len(test_cases)
    avg_correctness = sum(r['metrics'].get('answer_correctness', 0) for r in results) / len(test_cases)
    avg_relevance = sum(r['metrics'].get('retrieval_relevance', 0) for r in results) / len(test_cases)

    md_content = f"""# Báo cáo đánh giá hệ thống RAG (RAG Evaluation Report)

Báo cáo này minh họa phương pháp đánh giá hệ thống RAG (Đáp ứng tiêu chí III. Evaluation RAG Requirement).

## 1. Phương pháp đo lường (Measurement Approach)
- **Latency (Độ trễ):** Đo lường thời gian (mili-giây) từ lúc nhận câu hỏi đến khi trả về câu trả lời hoàn chỉnh.
- **Quantitative Metrics (Định lượng):** Áp dụng phương pháp **LLM-as-a-judge**. Sử dụng chính mô hình LLM để chấm điểm theo thang 1-5 cho 2 tiêu chí:
  - *Retrieval Relevance (Độ liên quan của dữ liệu truy xuất):* Dữ liệu vector lấy lên từ Qdrant có chứa thông tin trả lời cho câu hỏi không.
  - *Answer Correctness (Độ chính xác của câu trả lời):* Câu trả lời có đúng trọng tâm, không bịa đặt (hallucination) dựa trên context hay không.
- **Qualitative Metrics (Định tính):** LLM sinh ra lời giải thích (qualitative reasoning) cho điểm số nó đánh giá.

## 2. Kết quả tổng quan (Overall Performance)
- **Trung bình Độ trễ (Avg Latency):** {avg_latency:.2f} ms
- **Độ chính xác câu trả lời (Answer Correctness):** {avg_correctness:.1f} / 5.0
- **Độ liên quan dữ liệu (Retrieval Relevance):** {avg_relevance:.1f} / 5.0

## 3. Chi tiết các Test Case (Detailed Test Cases)

"""
    for i, r in enumerate(results, 1):
        md_content += f"""### Test Case {i}
**Q:** {r['question']}
- **Latency:** {r['latency_ms']} ms
- **Sources Retrieved:** {r['num_sources_retrieved']} đoạn văn bản.
- **Quantitative Scores:**
  - Answer Correctness: {r['metrics'].get('answer_correctness')}/5
  - Retrieval Relevance: {r['metrics'].get('retrieval_relevance')}/5
- **Qualitative Reasoning:** {r['metrics'].get('qualitative_reasoning')}
- **System Answer:** 
> {r['answer'].replace(chr(10), chr(10) + "> ")}

---
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"\nĐã xuất báo cáo đánh giá chi tiết ra file: {report_path}")

if __name__ == "__main__":
    run_evaluation()
