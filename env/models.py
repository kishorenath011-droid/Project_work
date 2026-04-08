from typing import Literal, Optional
from pydantic import BaseModel, Field


class Email(BaseModel):
    """
    Represents a single email in the environment.
    """
    id: str = Field(..., description="Unique identifier for the email")
    subject: str
    body: str
    sender: str

    # Optional metadata (can be extended later)
    category: Optional[str] = None  # e.g., 'spam', 'important', etc.


class Action(BaseModel):
    """
    Represents an action taken by the agent.
    """
    type: Literal[
        "classify",
        "summarize",
        "reply",
        "delete"
    ]

    # Optional payload depending on action type
    content: Optional[str] = None


class Task(BaseModel):
    """
    Represents the expected outcome for an email.
    Used by the grader.
    """
    email_id: str

    # What the agent is expected to do
    expected_action: Literal[
        "classify",
        "summarize",
        "reply",
        "delete"
    ]

    # Ground truth output (if applicable)
    expected_output: Optional[str] = None