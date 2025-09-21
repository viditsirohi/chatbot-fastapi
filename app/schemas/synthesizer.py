"""Synthesizer node structured output schema."""

from enum import Enum
from typing import List

from pydantic import (
    BaseModel,
    Field,
)


class ConfidenceLevel(str, Enum):
    """Confidence levels for synthesized responses."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SynthesisResponse(BaseModel):
    """Structured output for the synthesizer node."""
    
    response: str = Field(
        description="The complete synthesized response to the user's query - short, crisp, and in plain English"
    )