"""This file contains the chat schema for the application."""

import re
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
)
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class Message(BaseModel):
    """Message model for chat endpoint.

    Attributes:
        role: The role of the message sender (user or assistant).
        content: The content of the message.
        response_strategy: The bot's strategy context for maintaining conversation flow.
    """

    model_config = {"extra": "ignore"}

    role: Literal["user", "assistant", "system"] = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message", min_length=1, max_length=3000)
    response_strategy: Optional[str] = Field(
        default=None, 
        description="Simple strategy context for what the bot should do next to maintain conversation flow"
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate the message content.

        Args:
            v: The content to validate

        Returns:
            str: The validated content

        Raises:
            ValueError: If the content contains disallowed patterns
        """
        # Check for potentially harmful content
        if re.search(r"<script.*?>.*?</script>", v, re.IGNORECASE | re.DOTALL):
            raise ValueError("Content contains potentially harmful script tags")

        # Check for null bytes
        if "\0" in v:
            raise ValueError("Content contains null bytes")

        return v


class ChatRequest(BaseModel):
    """Request model for chat endpoint.

    Attributes:
        messages: List of messages in the conversation.
        session_id: The session ID for the conversation.
        profile: The archetype profile of the user.
        primary_archetype: The primary archetype of the user.
        secondary_archetype: The secondary archetype of the user.
    """

    messages: List[Message] = Field(
        ...,
        description="List of messages in the conversation",
        min_length=1,
    )
    session_id: str = Field(..., description="The session ID for the conversation")
    profile: str = Field(..., description="The archetype profile of the user")
    primary_archetype: str = Field(..., description="The primary archetype of the user")
    secondary_archetype: str = Field(..., description="The secondary archetype of the user")

class NotificationPayload(BaseModel):
    """Schema for notification scheduling payload."""
    
    should_schedule: bool = Field(
        default=False, 
        description="Whether a notification should be scheduled"
    )
    reminder_type: Optional[str] = Field(
        default=None, 
        description="Type of reminder (commitment, follow_up, check_in)"
    )
    frequency: Optional[str] = Field(
        default=None, 
        description="Reminder frequency (daily, weekly, monthly)"
    )
    date: Optional[str] = Field(
        default=None, 
        description="Specific date for reminder (YYYY-MM-DD)"
    )
    message: Optional[str] = Field(
        default=None, 
        description="Custom reminder message"
    )
    timezone: str = Field(
        default="Asia/Kolkata", 
        description="Timezone for reminder scheduling (default: India Standard Time)"
    )
    scheduled_time: str = Field(
        default="09:00", 
        description="Time of day for reminder (default: 9:00 AM)"
    )
    commitment_id: Optional[str] = Field(
        default=None,
        description="ID of the commitment this reminder is for"
    )


class ChatResponsePayload(BaseModel):
    """Additional payload data for chat responses."""
    
    notification: NotificationPayload = Field(
        default_factory=NotificationPayload,
        description="Notification scheduling information"
    )
    session_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional session state information"
    )
    coaching_stage: Optional[str] = Field(
        default=None,
        description="Current stage in coaching flow"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint.

    Attributes:
        messages: List of messages in the conversation.
        payload: Additional response data and state.
    """

    messages: List[Message] = Field(..., description="List of messages in the conversation")
    payload: ChatResponsePayload = Field(
        default_factory=ChatResponsePayload,
        description="Additional response data and state"
    )


class StreamResponse(BaseModel):
    """Response model for streaming chat endpoint.

    Attributes:
        content: The content of the current chunk.
        done: Whether the stream is complete.
    """

    content: str = Field(default="", description="The content of the current chunk")
    done: bool = Field(default=False, description="Whether the stream is complete")


class LogChatCreate(BaseModel):
    """Schema for creating a log_chat entry.

    Attributes:
        id: The session ID that becomes the primary key for the chat log.
        user_id: The ID of the user who participated in the chat.
        chat: The chat messages as a list of message objects.
        summary: Optional summary of the chat session.
    """

    id: str = Field(..., description="The session ID that becomes the primary key for the chat log")
    user_id: UUID = Field(..., description="The ID of the user who participated in the chat")
    chat: List[Dict[str, Any]] = Field(..., description="The chat messages as a list of message objects")
    summary: Optional[str] = Field(default=None, description="Optional summary of the chat session")


class LogChatResponse(BaseModel):
    """Schema for log_chat response.

    Attributes:
        id: The session ID that serves as the primary key for the chat log.
        created_at: When the log entry was created.
        user_id: The ID of the user who participated in the chat.
        chat: The chat messages as a list of message objects.
        summary: Optional summary of the chat session.
    """

    id: str = Field(..., description="The session ID that serves as the primary key for the chat log")
    created_at: datetime = Field(..., description="When the log entry was created")
    user_id: UUID = Field(..., description="The ID of the user who participated in the chat")
    chat: List[Dict[str, Any]] = Field(..., description="The chat messages as a list of message objects")
    summary: Optional[str] = Field(default=None, description="Optional summary of the chat session")
