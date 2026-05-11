from fastapi import APIRouter
from pydantic import BaseModel
from app.services.email_service import process_email, fetch_emails

router = APIRouter()


class EmailRequest(BaseModel):
    text: str


@router.post("/predict")
def predict(request: EmailRequest):
    return process_email(request.text)


# ✅ THIS MUST EXIST
@router.get("/emails")
def get_emails(label: str = None, limit: int = 10):
    return {"data": fetch_emails(label, limit)}