from fastapi import APIRouter

from app.models.schemas import AskRequest, AskResponse
from app.services.inference import generate_answer

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest) -> AskResponse:
    answer = generate_answer(payload.question)
    return AskResponse(answer=answer)
