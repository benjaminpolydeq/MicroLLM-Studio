from fastapi import APIRouter

router = APIRouter()

@router.post("/infer")
def infer(payload: dict):
    return {"result": f"Processed {payload}"}
