import chromadb
import json
from pathlib import Path

# Khởi tạo lưu trữ tại thư mục local 'data/vector_db'
client = chromadb.PersistentClient(path="./data/vector_db")
# Tạo collection chuyên biệt cho tri thức
collection = client.get_or_create_collection(name="language_center_knowledge")

INPUT_FOLDER = Path(r"data/.cache/integrated_knowledge_maps")

def ingest_data():
    json_files = list(INPUT_FOLDER.glob("*.json"))
    print(f"Tìm thấy {len(json_files)} file tri thức. Đang nạp vào ChromaDB...")

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            ids, documents, metadatas = [], [], []
            
            for item in data:
                # ID định dạng tenant:entity để đảm bảo tính Multi-tenant
                doc_id = f"{item['metadata']['tenant_id']}:{item['entity_id']}"
                ids.append(doc_id)
                documents.append(item['knowledge_cluster'])
                
                # Chuẩn hóa metadata (ChromaDB yêu cầu dict phẳng)
                meta = {
                    "tenant_id": item['metadata']['tenant_id'],
                    "entity_id": item['entity_id'],
                    "is_commercial": item['metadata']['is_commercial'],
                    "evidence": json.dumps(item['metadata']['evidence']), # Convert dict sang string
                    "related_partners": ", ".join(item['metadata']['related_partners'])
                }
                metadatas.append(meta)
            
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
            print(f"✅ Đã nạp {len(ids)} cụm tri thức từ {json_file.name}")

if __name__ == "__main__":
    ingest_data()