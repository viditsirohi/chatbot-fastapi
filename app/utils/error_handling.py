"""User-friendly error handling utilities.

This module provides utilities to convert technical errors into user-friendly messages
that don't expose internal system details to end users.
"""

from app.core.logging import logger


def get_user_friendly_error_message(
    error: Exception | str, 
    context: str = "general",
    user_id: str = None,
    session_id: str = None
) -> str:
    """Convert technical errors into user-friendly messages.
    
    Args:
        error: The exception or error message to convert
        context: Context of where the error occurred (tool, api, etc.)
        user_id: User ID for logging purposes
        session_id: Session ID for logging purposes
        
    Returns:
        str: User-friendly error message
    """
    # Log the technical error for debugging
    error_str = str(error)
    logger.error(
        f"user_friendly_error_conversion",
        original_error=error_str,
        context=context,
        user_id=user_id,
        session_id=session_id,
        exc_info=isinstance(error, Exception)
    )
    
    # Return a generic, friendly error message
    return (
        "I'm experiencing some technical difficulties right now. "
        "Let's get back to this later. Is there anything else I can help you with?"
    )


def handle_tool_error(
    error: Exception | str,
    tool_name: str,
    user_id: str = None,
    session_id: str = None
) -> str:
    """Handle tool execution errors with user-friendly messaging.
    
    Args:
        error: The exception or error message
        tool_name: Name of the tool that failed
        user_id: User ID for logging
        session_id: Session ID for logging
        
    Returns:
        str: User-friendly error message
    """
    return get_user_friendly_error_message(
        error=error,
        context=f"tool_{tool_name}",
        user_id=user_id,
        session_id=session_id
    )


def handle_api_error(
    error: Exception | str,
    endpoint: str,
    user_id: str = None,
    session_id: str = None
) -> str:
    """Handle API errors with user-friendly messaging.
    
    Args:
        error: The exception or error message
        endpoint: API endpoint that failed
        user_id: User ID for logging
        session_id: Session ID for logging
        
    Returns:
        str: User-friendly error message
    """
    return get_user_friendly_error_message(
        error=error,
        context=f"api_{endpoint}",
        user_id=user_id,
        session_id=session_id
    )


def is_user_facing_error(error_message: str) -> bool:
    """Check if an error message is safe to show to users.
    
    Some error messages are already user-friendly and can be shown directly.
    
    Args:
        error_message: The error message to check
        
    Returns:
        bool: True if the error is safe to show to users
    """
    user_friendly_patterns = [
        "You already have",  # Commitment limit errors
        "Please complete some existing",
        "Commitment text is required",
        "Commitment ID is required",
        "Invalid frequency",
        "Must specify either frequency or date",
        "Cannot specify both frequency and date",
        "No reminder" in error_message.lower(),
        "validation error" in error_message.lower() and "field" in error_message.lower()
    ]
    
    # Check if the error contains any user-friendly patterns
    for pattern in user_friendly_patterns:
        if pattern in error_message:
            return True
    
    # Check if it's a business logic error (doesn't contain technical terms)
    technical_terms = [
        "traceback", "exception", "stack", "error:", "failed:",
        "supabase", "database", "sql", "connection", "timeout",
        "authentication", "token", "jwt", "credentials", "401", "500",
        "internal", "server", "client", "http", "api", "endpoint"
    ]
    
    error_lower = error_message.lower()
    for term in technical_terms:
        if term in error_lower:
            return False
    
    return True

