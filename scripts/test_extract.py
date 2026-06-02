import json
from app.core.bootstrap import bootstrap_runtime
from llama_index.core import Settings

bootstrap_runtime()

questions = [
    "Chọn ra 5 người hợp với role senior AI Engineer (kinh nghiệm trên 3 năm)",
    "Top 3 ứng viên pytohn"
]

for q in questions:
    extract_prompt = (
        "Bạn là trợ lý Nhân sự. Hãy trích xuất yêu cầu tìm kiếm ứng viên từ câu hỏi sau.\n"
        "Chỉ trả về JSON hợp lệ với 2 trường:\n"
        "- skills: mảng các chuỗi (ví dụ: ['Python', 'AI'])\n"
        "- min_experience: số nguyên (năm kinh nghiệm tối thiểu, mặc định là 0)\n\n"
        f"Câu hỏi: {q}\n\n"
        "JSON:"
    )
    resp = Settings.llm.complete(extract_prompt)
    print(f"Q: {q}")
    print(f"RAW: {resp.text}")
