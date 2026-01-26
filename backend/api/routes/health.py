from fastapi import APIRouter, HTTPException
from infrastructure.db.connection import get_connection

router = APIRouter()

@router.get("/live")
def live():
    return {"status": "OK"}

@router.get("/health")
def health():
    try:
        con = get_connection()
        try:
            con.execute("SELECT 1")
        finally:
            con.close()
        return {"status": "OK"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")