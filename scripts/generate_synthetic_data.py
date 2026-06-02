import os
import random
import datetime

# Configuration
OUTPUT_DIR = os.path.join("data", "knowledge_base", "synthetic")
NUM_FILES = 20

# Data pools
CENTER_PREFIXES = ["Trung tâm Anh ngữ", "Học viện Ngôn ngữ", "Trường Ngoại ngữ", "English Center", "Academy"]
CENTER_NAMES = [
    "StarLight", "FutureBright", "GlobalConnect", "VisaPrep", "MasterTalk", 
    "ElitePath", "Sunshine", "OceanBlue", "MountainTop", "NextGen",
    "Pioneer", "Summit", "Vertex", "Horizon", "Zenith", "Apex",
    "Focus", "Target", "Goal", "Success"
]
LOCATIONS = ["Quận 1", "Quận 3", "Quận 10", "Cầu Giấy", "Ba Đình", "Hải Châu", "Bình Thạnh", "Thủ Đức"]
STREETS = ["Nguyễn Huệ", "Lê Lợi", "Cách Mạng Tháng 8", "Xuân Thủy", "Kim Mã", "Lê Duẩn", "Phạm Văn Đồng"]

COURSES = [
    {"name": "IELTS Foundation", "target": "4.5-5.0"},
    {"name": "IELTS Intensive", "target": "6.5+"},
    {"name": "TOEIC Basic", "target": "450+"},
    {"name": "TOEIC Advanced", "target": "800+"},
    {"name": "Giao tiếp Phản xạ", "target": "N/A"},
    {"name": "Tiếng Anh Thiếu nhi", "target": "Starters/Movers"},
    {"name": "Business English", "target": "Professional"},
]

POLICIES = [
    "Hoàn 100% học phí nếu không đạt đầu ra.",
    "Bảo lưu tối đa 6 tháng.",
    "Bảo lưu 3 tháng, phí 500k.",
    "Học lại miễn phí nếu thi trượt.",
    "Không hỗ trợ hoàn phí sau khai giảng.",
    "Giảm 10% cho nhóm 2 người.",
    "Tặng giáo trình gốc trị giá 300k.",
]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_vn_phone():
    prefixes = ["090", "091", "098", "097", "032", "070"]
    return f"{random.choice(prefixes)}{random.randint(1000000, 9999999)}"

def format_price(amount):
    # Noise injection: different price formats
    style = random.choice(["standard", "short", "nospace", "text"])
    if style == "standard":
        return f"{amount:,} VND".replace(",", ".")
    elif style == "short":
        return f"{amount//1000}k"
    elif style == "nospace":
        return f"{amount}d"
    else:
        return f"{amount/1000000} triệu"

def generate_file_content(index):
    center_name = f"{random.choice(CENTER_PREFIXES)} {random.choice(CENTER_NAMES)} {index}"
    address = f"Số {random.randint(1, 999)} {random.choice(STREETS)}, {random.choice(LOCATIONS)}"
    phone = generate_vn_phone()
    
    # Structure Simulation
    structure_type = random.choice(["structured", "plain", "messy"])
    
    content = []
    
    # Header
    if structure_type == "structured":
        content.append(f"# {center_name}\n")
        content.append(f"**Địa chỉ:** {address}")
        content.append(f"**Hotline:** {phone}\n")
    else:
        content.append(f"CHÀO MỪNG ĐẾN VỚI {center_name.upper()}")
        content.append(f"LH: {phone} - CS: {address}\n")

    # Courses
    content.append("## Các khóa học và Học phí\n" if structure_type == "structured" else "LIST KHÓA HỌC:\n")
    
    selected_courses = random.sample(COURSES, k=random.randint(3, 6))
    
    if structure_type == "messy":
        # Simulate bad OCR table
        content.append("| Khóa | Giá | Time |")
        content.append("|---|---|---|")
    
    for course in selected_courses:
        price_val = random.randint(30, 150) * 100000
        price_str = format_price(price_val)
        duration = random.randint(2, 6)
        
        if structure_type == "structured":
            content.append(f"### {course['name']}")
            content.append(f"- Mục tiêu: {course['target']}")
            content.append(f"- Thời lượng: {duration} tháng")
            content.append(f"- Học phí: {price_str}")
            content.append(f"- Khai giảng: {random.randint(1, 30)}/{random.randint(1, 12)}\n")
        elif structure_type == "messy":
             content.append(f"| {course['name']} | {price_str} | {duration}mo |")
        else:
            content.append(f"* {course['name']} ({course['target']}) - {duration} tháng - Giá: {price_str}")

    # Policies
    content.append("\n## Chính sách quy định" if structure_type == "structured" else "\nQUY ĐỊNH CHUNG")
    selected_policies = random.sample(POLICIES, k=2)
    for p in selected_policies:
        content.append(f"- {p}")

    return "\n".join(content)

def main():
    ensure_dir(OUTPUT_DIR)
    print(f"Generating {NUM_FILES} synthetic files in {OUTPUT_DIR}...")
    
    for i in range(1, NUM_FILES + 1):
        content = generate_file_content(i)
        tenant_id = f"synthetic_{i:02d}"
        filename = f"tenant_{tenant_id}.md"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
    print("Done generating synthetic data.")

if __name__ == "__main__":
    main()
