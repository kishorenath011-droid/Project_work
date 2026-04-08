from pydantic import BaseModel, Field
from typing import Literal, Optional

class Task(BaseModel):
    """
    Represents a task the agent must complete for a specific email.
    """
    email_id: str = Field(..., description="Unique identifier for the email")
    expected_action: Literal[
        "classify",
        "summarize",
        "reply",
        "delete"
    ] = Field(..., description="The action the agent must take")
    expected_output: Optional[str] = Field(None, description="The expected content or output, if relevant")