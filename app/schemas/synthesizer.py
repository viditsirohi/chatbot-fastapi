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
        description="The complete synthesized response to the user's query"
    )
    
    confidence_level: ConfidenceLevel = Field(
        description="Confidence level in the response based on available information"
    )
    
    sources_used: List[str] = Field(
        default_factory=list,
        description="List of sources/tools that contributed to this response"
    )
    
    limitations: List[str] = Field(
        default_factory=list,
        description="Any limitations or caveats in the response"
    )
    
    follow_up_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions the user might ask"
    )
