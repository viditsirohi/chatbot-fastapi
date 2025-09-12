"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat,
streaming chat, message history management, and chat history clearing.
"""

import json
from typing import List

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse

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

router = APIRouter()
agent = LangGraphAgent()



@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat"][0])
async def chat(
    request: Request,
    chat_request: ChatRequest,
    user: dict = Depends(get_current_user),
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

        result = await agent.get_response(
            chat_request.messages, 
            chat_request.session_id, 
            user_id=user["id"],
            profile=chat_request.profile,
            primary_archetype=chat_request.primary_archetype,
            secondary_archetype=chat_request.secondary_archetype,
        )

        logger.info("chat_request_processed", session_id=chat_request.session_id, user_id=user["id"])

        return ChatResponse(messages=result)
    except Exception as e:
        logger.error("chat_request_failed", session_id=chat_request.session_id, user_id=user["id"], error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat_stream"][0])
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    user: dict = Depends(get_current_user),
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
                        profile=chat_request.profile,
                        primary_archetype=chat_request.primary_archetype,
                        secondary_archetype=chat_request.secondary_archetype,
                     ):
                        full_response += chunk
                        response = StreamResponse(content=chunk, done=False)
                        yield f"data: {json.dumps(response.model_dump())}\n\n"

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
                error_response = StreamResponse(content=str(e), done=True)
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
        raise HTTPException(status_code=500, detail=str(e))


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
