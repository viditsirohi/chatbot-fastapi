"""Tool parameter schemas for type safety and validation.

This module defines Pydantic models for all tool parameters to ensure
proper validation and type checking for LangGraph tools.
"""

from typing import Optional

from pydantic import (
    BaseModel,
    Field,
)


class BaseToolParams(BaseModel):
    """Base parameters for all authenticated tools."""
    user_id: Optional[str] = Field(default=None, description="The authenticated user's ID (injected at runtime)")
    access_token: Optional[str] = Field(default=None, description="The user's access token (injected at runtime)")


class CommitmentFetchParams(BaseToolParams):
    """Parameters for fetching user commitments."""
    pass


class CommitmentCreateParams(BaseToolParams):
    """Parameters for creating user commitments."""
    commitment_text: str = Field(..., description="The specific commitment text the user wants to set", min_length=1)
    chat_id: str = Field(..., description=" chat session ID for tracking")
    reminder_set: bool = Field(default=False, description="Whether to mark that a reminder is set")


class CommitmentCompleteParams(BaseToolParams):
    """Parameters for completing user commitments."""
    commitment_id: str = Field(..., description="ID of the commitment to mark as complete", min_length=1)


class JournalFetchParams(BaseToolParams):
    """Parameters for fetching user journal entries."""
    pass


class ReminderFetchParams(BaseToolParams):
    """Parameters for fetching user reminders."""
    pass


class ReminderSetParams(BaseToolParams):
    """Parameters for setting user reminders."""
    frequency: Optional[str] = Field(default=None, description="Reminder frequency (daily, weekly, monthly, etc.)")
    date: Optional[str] = Field(default=None, description="Specific date for the reminder (YYYY-MM-DD format)")
    chat_id: str = Field(..., description=" chat session ID for tracking")


class ReminderUpdateParams(BaseToolParams):
    """Parameters for updating user reminders."""
    reminder_id: str = Field(..., description="ID of the reminder to update", min_length=1)
    frequency: Optional[str] = Field(default=None, description="New reminder frequency (daily, weekly, monthly, etc.)")
    date: Optional[str] = Field(default=None, description="New specific date for the reminder (YYYY-MM-DD format)")


class CommitmentReminderOfferParams(BaseToolParams):
    """Parameters for offering commitment reminders."""
    commitment_text: str = Field(..., description="The commitment text for context", min_length=1)
    commitment_id: str = Field(..., description="ID of the commitment to remind about", min_length=1)
    chat_id: str = Field(..., description=" chat session ID for tracking")


class CommitmentReminderSetParams(BaseToolParams):
    """Parameters for setting commitment reminders."""
    reminder_type: str = Field(..., description="Type of reminder ('frequency' or 'date')")
    commitment_id: str = Field(..., description="ID of the commitment to remind about", min_length=1)
    frequency: Optional[str] = Field(default=None, description="Reminder frequency (daily, weekly, monthly) - if reminder_type is 'frequency'")
    date: Optional[str] = Field(default=None, description="Specific date (YYYY-MM-DD) - if reminder_type is 'date'")
    chat_id: str = Field(..., description=" chat session ID for tracking")


class CommitmentReminderDeclineParams(BaseToolParams):
    """Parameters for declining commitment reminders."""
    commitment_id: str = Field(..., description="ID of the commitment", min_length=1)
