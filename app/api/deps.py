import os

try:
    import firebase_admin
    from firebase_admin import auth, credentials

    _FIREBASE_IMPORT_ERROR = None
except Exception as _e:
    firebase_admin = None
    auth = None
    credentials = None
    _FIREBASE_IMPORT_ERROR = _e
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import ALLOW_UID_AS_TENANT_FALLBACK


def _ensure_firebase_app() -> None:
    """
    Lazy init to avoid import-time crash when service account is missing.
    Configure via env:
      FIREBASE_SERVICE_ACCOUNT_PATH=.../service_account.json
    Defaults to `service_account.json` at repo root.
    """
    if _FIREBASE_IMPORT_ERROR is not None or firebase_admin is None or credentials is None:
        raise RuntimeError(
            "Missing dependency `firebase_admin`. Install Firebase Admin SDK to use /chat endpoint."
        )

    if firebase_admin._apps:
        return
    path = (os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH") or "service_account.json").strip()
    if not path or not os.path.exists(path):
        raise RuntimeError(
            "Missing Firebase service account file. Set FIREBASE_SERVICE_ACCOUNT_PATH or place `service_account.json` in repo root."
        )
    cred = credentials.Certificate(path)
    firebase_admin.initialize_app(cred)


security = HTTPBearer()


def get_current_user(token: HTTPAuthorizationCredentials = Depends(security)):
    try:
        _ensure_firebase_app()
        decoded_token = auth.verify_id_token(token.credentials)  # type: ignore[union-attr]

        user_id = decoded_token["uid"]
        tenant_id = decoded_token.get("tenant_id")

        # Strict by default in production; can be enabled in dev with env.
        if not tenant_id and ALLOW_UID_AS_TENANT_FALLBACK:
            tenant_id = user_id
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Missing tenant_id claim in token.",
            )

        return {
            "uid": user_id,
            "tenant_id": tenant_id,
            "email": decoded_token.get("email"),
        }
    except HTTPException:
        raise
    except RuntimeError as e:
        # Configuration error (missing service account, etc.) should be a 500, not a 401.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token không hợp lệ hoặc đã hết hạn",
        )
