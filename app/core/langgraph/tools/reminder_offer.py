"""Reminder offering tool for commitment follow-up.

This module provides functionality to offer and set reminders after commitment creation,
with automatic notification payload generation for frontend device scheduling.
"""

from typing import (
    Dict,
    Optional,
)

from langchain_core.tools import tool

from app.core.logging import logger
from app.schemas.chat import NotificationPayload

from .reminder_manage import _reminder_manager
from .reminder_validation import (
    normalize_frequency,
    validate_reminder_data,
)


@tool
async def offer_commitment_reminder(
    commitment_text: str,
    commitment_id: str,
    user_id: str = None,
    access_token: str = None,
    chat_id: str = None
) -> str:
    """Offer to set up a reminder for a newly created commitment.
    
    This tool should be used after a commitment is successfully created to offer
    reminder options to the user. It suggests contextual timing based on the commitment.
    
    Args:
        commitment_text: The commitment text for context
        commitment_id: ID of the commitment to remind about
        user_id: The authenticated user's ID (will be injected at runtime)
        access_token: The user's access token (will be injected at runtime)
        chat_id: Optional chat session ID for tracking
        
    Returns:
        str: Reminder offer message with options
    """
    # Validate required parameters
    if not commitment_text or not commitment_text.strip():
        return "Error: Commitment text is required and cannot be empty"
    if not commitment_id or not commitment_id.strip():
        return "Error: Commitment ID is required and cannot be empty"
    
    logger.info(
        "reminder_offer_initiated",
        user_id=user_id,
        commitment_id=commitment_id,
        has_commitment_text=bool(commitment_text)
    )
    
    # Create contextual reminder offer
    offer_message = f"""Great! Your commitment is set: "{commitment_text}"

Would you like me to set up a reminder to help you stay on track? I can remind you:

1. **Daily** - Every day at 9 AM
2. **Weekly** - Every week on a specific day at 9 AM
3. **Fortnightly** - Every two weeks at 9 AM
4. **Monthly** - Every month at 9 AM
5. **Specific date** - On a particular date at 9 AM

All reminders will be sent at 9 AM India time. What would work best for you?

Or say "no reminder" if you prefer to track it yourself."""
    
    return offer_message


@tool 
async def set_commitment_reminder(
    reminder_type: str,
    commitment_id: str,
    frequency: str = None,
    date: str = None,
    user_id: str = None,
    access_token: str = None,
    chat_id: str = None
) -> str:
    """Set a reminder for a commitment and generate notification payload.
    
    This tool sets up a reminder in the database and returns a notification payload
    for the frontend to schedule device notifications.
    
    Args:
        reminder_type: Type of reminder ("frequency" or "date")
        commitment_id: ID of the commitment to remind about
        frequency: Reminder frequency (daily, weekly, monthly) - if reminder_type is "frequency"
        date: Specific date (YYYY-MM-DD) - if reminder_type is "date"
        user_id: The authenticated user's ID (will be injected at runtime)
        access_token: The user's access token (will be injected at runtime)
        chat_id: Optional chat session ID for tracking
        
    Returns:
        str: Success message with notification payload info
    """
    # Validate required parameters
    if not reminder_type or not reminder_type.strip():
        return "Error: Reminder type is required and cannot be empty"
    if not commitment_id or not commitment_id.strip():
        return "Error: Commitment ID is required and cannot be empty"
    
    try:
        # Validate input
        if reminder_type not in ["frequency", "date"]:
            return "Error: Reminder type must be either 'frequency' or 'date'"
        
        if reminder_type == "frequency" and not frequency:
            return "Error: Frequency is required when reminder_type is 'frequency'"
        
        if reminder_type == "date" and not date:
            return "Error: Date is required when reminder_type is 'date'"
        
        if reminder_type == "frequency" and reminder_type == "date":
            return "Error: Cannot set both frequency and date - choose one"
        
        # Normalize and validate frequency/date values
        normalized_frequency = normalize_frequency(frequency) if frequency else None
        validation_error = validate_reminder_data(normalized_frequency, date)
        
        if validation_error:
            return f"Error: {validation_error}"
        
        # Set reminder in database using existing functionality with normalized values
        result = await _reminder_manager.set_reminder(
            user_id=user_id,
            access_token=access_token,
            frequency=normalized_frequency if reminder_type == "frequency" else None,
            date=date if reminder_type == "date" else None,
            chat_id=chat_id
        )
        
        # Only generate payload if reminder was actually successfully set
        if "successfully set" not in result:
            logger.info(
                "commitment_reminder_failed",
                user_id=user_id,
                commitment_id=commitment_id,
                error=result
            )
            return result  # Return error message from reminder manager (no payload)
        
        # Create notification payload info ONLY for successful reminder creation
        notification_info = {
            "should_schedule": True,
            "reminder_type": "commitment",
            "timezone": "Asia/Kolkata",
            "scheduled_time": "09:00",
            "commitment_id": commitment_id,
            "message": f"Time to work on your commitment!"
        }
        
        if reminder_type == "frequency":
            notification_info["frequency"] = normalized_frequency
            success_msg = f"✅ Perfect! I've set up a {normalized_frequency} reminder for your commitment at 9 AM India time."
        else:
            notification_info["date"] = date
            success_msg = f"✅ Perfect! I've set up a reminder for {date} at 9 AM India time for your commitment."
        
        logger.info(
            "commitment_reminder_set_success",
            user_id=user_id,
            commitment_id=commitment_id,
            reminder_type=reminder_type,
            frequency=normalized_frequency,
            date=date
        )
        
        # Add payload information to the response ONLY after confirming success
        success_msg += f"\n\nNotification scheduled - you'll receive reminders on your device to help you stay committed!"
        success_msg += f"\n\n[NOTIFICATION_PAYLOAD: {notification_info}]"
        
        return success_msg
        
    except Exception as e:
        error_message = f"Error setting commitment reminder: {str(e)}"
        logger.error(
            "commitment_reminder_error",
            user_id=user_id if user_id else 'unknown',
            commitment_id=commitment_id,
            error=str(e)
        )
        return error_message


@tool
async def decline_commitment_reminder(
    commitment_id: str,
    user_id: str = None
) -> str:
    """Handle when user declines reminder setup for a commitment.
    
    Args:
        commitment_id: ID of the commitment 
        user_id: The authenticated user's ID (will be injected at runtime)
        
    Returns:
        str: Acknowledgment message
    """
    # Validate required parameters
    if not commitment_id or not commitment_id.strip():
        return "Error: Commitment ID is required and cannot be empty"
    logger.info(
        "commitment_reminder_declined",
        user_id=user_id,
        commitment_id=commitment_id
    )
    
    return "No problem! You can always set up a reminder later if you change your mind. Your commitment is saved and ready to track on your home screen."
