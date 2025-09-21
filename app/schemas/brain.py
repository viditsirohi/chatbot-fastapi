"""Brain node structured output schema."""

from enum import Enum
from typing import Optional

from pydantic import (
    BaseModel,
    Field,
)


class ComplexityLevel(str, Enum):
    """Complexity levels for user queries."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class BrainDecision(BaseModel):
    """Structured output for the brain node decisions."""
    
    complexity_level: ComplexityLevel = Field(
        description="Assessment of query complexity: simple, moderate, or complex"
    )
    
    needs_tools: bool = Field(
        description="Whether external tools/search are needed to answer the query"
    )
    
    needs_synthesis: bool = Field(
        description="Whether the query requires complex synthesis/analysis"
    )
    
    reasoning: str = Field(
        description="Brief explanation of the decision rationale"
    )
    
    direct_response: Optional[str] = Field(
        default=None,
        description="Complete direct answer for simple queries (null if tools/synthesis needed)"
    )
    
    tool_guidance: Optional[str] = Field(
        default=None,
        description="Guidance for what information to search for (when tools are needed)"
    )
    
    response_strategy: str = Field(
        description="Simple strategy for what should happen next in this conversation, using easy words and short sentences"
    )