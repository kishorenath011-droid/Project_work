from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal
from datetime import datetime

class Email(BaseModel):
    id: str = Field(..., description="Unique identifier for the email")
    subject: str
    body: str
    sender: str
    category: Optional[str] = None  # e.g., 'spam', 'important'
    timestamp: datetime = Field(default_factory=datetime.now)

class Action(BaseModel):
    type: Literal["classify", "summarize", "reply", "delete"]
    content: Optional[str] = None

class Task(BaseModel):
    email_id: str
    expected_action: Literal["classify", "summarize", "reply", "delete"]
    expected_output: Optional[str] = None

# You can add any additional models if needed, and this is now a standalone module