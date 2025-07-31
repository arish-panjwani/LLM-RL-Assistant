from pydantic import BaseModel
from typing import Optional

class PromptRequest(BaseModel):
    prompt: str
    form: str  # base64 image
    model: str

class PromptResponse(BaseModel):
    id: int
    code: int
    response: str
    sentiment: str

class FeedbackRequest(BaseModel):
    id: int
    text: str
    score: int
    model: str

class FeedbackResponse(BaseModel):
    code: int

class HistoryQuery(BaseModel):
    id: Optional[int] = None
