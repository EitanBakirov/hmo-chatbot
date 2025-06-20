from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ChatRequest(BaseModel):
    user_info: Dict = Field(default_factory=dict)
    history: List[Dict] = Field(default_factory=list)
    question: str
    language: str = "he"
    phase: str = "collection"  # Added phase field with default value
    hmo: Optional[str] = None
    tier: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    phase_transition: Optional[bool] = False
    user_info: Optional[Dict[str, Any]] = None
