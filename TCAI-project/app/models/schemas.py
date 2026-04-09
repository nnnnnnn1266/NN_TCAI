from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question about turtle care.")


class AskResponse(BaseModel):
    answer: str
