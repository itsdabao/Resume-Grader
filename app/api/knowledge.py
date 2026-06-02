from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid
import firebase_admin
from firebase_admin import storage
from app.api.deps import get_current_user
# Import database session (tùy setup của bạn, ví dụ get_db)
# from app.db.session import get_db 
# from app.models import Document (Model bạn cần tạo trong code Python tương ứng bảng SQL vừa chạy)

router = APIRouter()

@router.post("/knowledge/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user), # Bảo vệ API & lấy tenant_id
    # db: Session = Depends(get_db)
):
    tenant_id = current_user["tenant_id"]
    file_id = str(uuid.uuid4())
    
    # 1. Định nghĩa đường dẫn file trên Firebase Storage: tenants/{tenant_id}/{file_id}.pdf
    # Cách này giúp cô lập dữ liệu file vật lý tuyệt đối [cite: 133]
    bucket = storage.bucket() # Lấy bucket mặc định
    blob_path = f"tenants/{tenant_id}/docs/{file.filename}"
    blob = bucket.blob(blob_path)

    try:
        # 2. Upload file lên Cloud
        blob.upload_from_file(file.file, content_type=file.content_type)
        
        # 3. Tạo public url (hoặc signed url) để lưu vào DB
        # file_url = blob.public_url 

        # 4. Lưu thông tin vào Database (Postgres)
        # new_doc = Document(
        #     id=file_id,
        #     tenant_id=tenant_id,
        #     filename=file.filename,
        #     file_path=blob_path,
        #     status="PENDING"
        # )
        # db.add(new_doc)
        # db.commit()

        return {
            "message": "Upload thành công",
            "doc_id": file_id,
            "path": blob_path,
            "status": "PENDING"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi upload: {str(e)}")