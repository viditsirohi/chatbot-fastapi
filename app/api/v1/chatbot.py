"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat,
streaming chat, message history management, and chat history clearing.
"""

import json
from datetime import datetime
from typing import List

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from app.api.v1.auth import get_current_user
from app.core.config import settings
from app.core.langgraph.graph import LangGraphAgent
from app.core.limiter import limiter
from app.core.logging import logger
from app.core.metrics import llm_stream_duration_seconds
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    Message,
    StreamResponse,
)
from app.services.database import database_service
from app.utils.error_handling import handle_api_error

router = APIRouter()
agent = LangGraphAgent()
security = HTTPBearer()


def _validate_final_payload(notification_data: dict, messages: list) -> bool:
    """Final validation to ensure payload should be included in response.
    
    Args:
        notification_data: The notification payload data
        messages: The response messages
        
    Returns:
        bool: True if payload should be included, False otherwise
    """
    # Must have should_schedule = True
    if not notification_data.get("should_schedule", False):
        return False
    
    # Must have reminder_type
    if not notification_data.get("reminder_type"):
        return False
    
    # Must have either frequency or date (but not both)
    has_frequency = bool(notification_data.get("frequency"))
    has_date = bool(notification_data.get("date"))
    
    if not (has_frequency or has_date) or (has_frequency and has_date):
        return False
    
    # Check that the latest message contains confirmation language
    if messages:
        latest_message = messages[-1] if isinstance(messages, list) else messages
        content = latest_message.get("content", "") if isinstance(latest_message, dict) else getattr(latest_message, "content", "")
        
        confirmation_keywords = ["✅", "Perfect", "reminder", "set", "scheduled", "successfully"]
        has_confirmation = any(keyword.lower() in content.lower() for keyword in confirmation_keywords)
        
        if not has_confirmation:
            return False
    
    return True



@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat"][0])
async def chat(
    request: Request,
    chat_request: ChatRequest,
    user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Process a chat request using LangGraph.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        user: The current user from Supabase token.

    Returns:
        ChatResponse: The processed chat response.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        logger.info(
            "chat_request_received",
            session_id=chat_request.session_id,
            user_id=user["id"],
            message_count=len(chat_request.messages),
            profile=chat_request.profile,
            primary_archetype=chat_request.primary_archetype,
            secondary_archetype=chat_request.secondary_archetype,
        )

        messages, notification_data = await agent.get_response(
            chat_request.messages, 
            chat_request.session_id, 
            user_id=user["id"],
            access_token=credentials.credentials,
            profile=chat_request.profile,
            primary_archetype=chat_request.primary_archetype,
            secondary_archetype=chat_request.secondary_archetype,
        )

        logger.info("chat_request_processed", session_id=chat_request.session_id, user_id=user["id"])

        # Create response with notification payload if available
        from uuid import UUID

        from app.schemas.chat import (
            ChatResponsePayload,
            NotificationPayload,
        )
        
        payload = ChatResponsePayload()
        
        # Double-check payload legitimacy before including in response
        if notification_data and _validate_final_payload(notification_data, messages):
            payload.notification = NotificationPayload(**notification_data)
            logger.info("notification_payload_included", payload=notification_data)
        else:
            if notification_data:
                logger.info("notification_payload_excluded", reason="Failed final validation")

        # Prepare response
        response = ChatResponse(messages=messages, payload=payload)
        
        # Log chat interaction to log_chat table
        try:
            # Create combined messages list (request + response)
            # all_messages = [msg.model_dump() for msg in chat_request.messages]
            all_messages = [msg.model_dump() for msg in messages]
            # Add all response messages
            # all_messages.extend([])
            
            # Log the interaction (this runs asynchronously and won't block the response)
            await database_service.log_chat_interaction(
                session_id=chat_request.session_id,
                user_id=UUID(user["id"]),
                chat_data=all_messages,
                summary=None,  # Could be enhanced later with AI-generated summaries
                access_token=credentials.credentials
            )
        except Exception as e:
            # Log error but don't fail the main request
            logger.error("chat_logging_failed", session_id=chat_request.session_id, user_id=user["id"], error=str(e))
        
        return response
    except Exception as e:
        logger.error("chat_request_failed", session_id=chat_request.session_id, user_id=user["id"], error=str(e), exc_info=True)
        user_friendly_error = handle_api_error(
            error=e,
            endpoint="chat",
            user_id=user["id"],
            session_id=chat_request.session_id
        )
        raise HTTPException(status_code=500, detail=user_friendly_error)


@router.post("/chat/stream")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat_stream"][0])
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Process a chat request using LangGraph with streaming response.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        user: The current user from Supabase token.

    Returns:
        StreamingResponse: A streaming response of the chat completion.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        logger.info(
            "stream_chat_request_received",
            session_id=chat_request.session_id,
            user_id=user["id"],
            message_count=len(chat_request.messages),
            profile=chat_request.profile,
            primary_archetype=chat_request.primary_archetype,
            secondary_archetype=chat_request.secondary_archetype,
        )

        async def event_generator():
            """Generate streaming events.

            Yields:
                str: Server-sent events in JSON format.

            Raises:
                Exception: If there's an error during streaming.
            """
            try:
                full_response = ""
                with llm_stream_duration_seconds.labels(model=settings.LLM_MODEL).time():
                    async for chunk in agent.get_stream_response(
                        chat_request.messages, 
                        chat_request.session_id, 
                        user_id=user["id"],
                        access_token=credentials.credentials,
                        profile=chat_request.profile,
                        primary_archetype=chat_request.primary_archetype,
                        secondary_archetype=chat_request.secondary_archetype,
                     ):
                        full_response += chunk
                        response = StreamResponse(content=chunk, done=False)
                        yield f"data: {json.dumps(response.model_dump())}\n\n"

                # Log streaming chat interaction to log_chat table
                try:
                    from uuid import UUID

                    # Create combined messages list (request + response)
                    all_messages = [msg.model_dump() for msg in chat_request.messages]
                    # Add the assistant's response
                    all_messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "response_strategy": None
                    })
                    
                    # Log the interaction (this runs asynchronously and won't block the response)
                    await database_service.log_chat_interaction(
                        session_id=chat_request.session_id,
                        user_id=UUID(user["id"]),
                        chat_data=all_messages,
                        summary=None,  # Could be enhanced later with AI-generated summaries
                        access_token=credentials.credentials
                    )
                except Exception as log_error:
                    # Log error but don't fail the main request
                    logger.error("stream_chat_logging_failed", session_id=chat_request.session_id, user_id=user["id"], error=str(log_error))

                # Send final message indicating completion
                final_response = StreamResponse(content="", done=True)
                yield f"data: {json.dumps(final_response.model_dump())}\n\n"

            except Exception as e:
                logger.error(
                    "stream_chat_request_failed",
                    session_id=chat_request.session_id,
                    user_id=user["id"],
                    error=str(e),
                    exc_info=True,
                )
                user_friendly_error = handle_api_error(
                    error=e,
                    endpoint="chat_stream_generator",
                    user_id=user["id"],
                    session_id=chat_request.session_id
                )
                error_response = StreamResponse(content=user_friendly_error, done=True)
                yield f"data: {json.dumps(error_response.model_dump())}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(
            "stream_chat_request_failed",
            session_id=chat_request.session_id,
            user_id=user["id"],
            error=str(e),
            exc_info=True,
        )
        user_friendly_error = handle_api_error(
            error=e,
            endpoint="chat_stream",
            user_id=user["id"],
            session_id=chat_request.session_id
        )
        raise HTTPException(status_code=500, detail=user_friendly_error)


@router.get("/messages", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def get_session_messages(
    request: Request,
    chat_request: ChatRequest,
    user: dict = Depends(get_current_user),
):
    """Get all messages for a session.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        user: The current user from Supabase token.

    Returns:
        ChatResponse: All messages in the session.

    Raises:
        HTTPException: If there's an error retrieving the messages.
    """
    try:
        messages = await agent.get_chat_history(chat_request.session_id)
        return ChatResponse(messages=messages)
    except Exception as e:
        logger.error("get_messages_failed", session_id=chat_request.session_id, user_id=user["id"], error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/messages")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def clear_chat_history(
    request: Request,
    chat_request: ChatRequest,
    user: dict = Depends(get_current_user),
):
    """Clear all messages for a session.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        user: The current user from Supabase token.

    Returns:
        dict: A message indicating the chat history was cleared.
    """
    try:
        await agent.clear_chat_history(chat_request.session_id)
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        logger.error("clear_chat_history_failed", session_id=chat_request.session_id, user_id=user["id"], error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
