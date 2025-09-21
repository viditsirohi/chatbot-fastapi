"""Reminder management tool for LangGraph.

This module provides tools for setting, updating, and managing user reminders
in the Supabase log_reminder table. This allows users to set notifications
for their commitments and other important events.
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from langchain_core.tools import tool
from supabase import Client

from app.core.logging import logger

from .base_supabase_tool import BaseSupabaseTool
from .reminder_validation import (
    normalize_frequency,
    validate_reminder_data,
)


class ReminderManager(BaseSupabaseTool):
    """Tool for managing user reminders in the log_reminder table."""
    
    def __init__(self):
        super().__init__(table_name="log_reminder", tool_name="reminder_manager")
    
    def _build_query(self, client: Client, user_id: str) -> Any:
        """Build query to fetch reminders for the user."""
        return client.table("log_reminder").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    
    def _format_results(self, data: List[Dict], user_id: str) -> Dict[str, Any]:
        """Format reminders data for AI consumption."""
        formatted_reminders = []
        for reminder in data:
            formatted_reminders.append({
                "id": reminder.get("id"),
                "frequency": reminder.get("frequency"),
                "date": reminder.get("date"),
                "user_id": reminder.get("user_id"),
                "chat_id": reminder.get("chat_id"),
                "created_at": reminder.get("created_at")
            })
        
        return {
            "user_id": user_id,
            "total_reminders": len(formatted_reminders),
            "reminders": formatted_reminders
        }
    
    async def set_reminder(
        self, 
        user_id: str, 
        access_token: str,
        frequency: Optional[str] = None,
        date: Optional[str] = None,
        chat_id: Optional[str] = None
    ) -> str:
        """Set a reminder for the user using enhanced base functionality.
        
        Args:
            user_id: The authenticated user's ID
            access_token: The user's access token
            frequency: Reminder frequency (daily, weekly, monthly, etc.)
            date: Specific date for the reminder (YYYY-MM-DD format)
            chat_id: Optional chat session ID
            
        Returns:
            str: Success message or error message
        """
        # Normalize and validate input
        normalized_frequency = normalize_frequency(frequency) if frequency else None
        validation_error = validate_reminder_data(normalized_frequency, date)
        
        if validation_error:
            return f"Error: {validation_error}"
        
        reminder_data = {}
        
        if normalized_frequency:
            reminder_data["frequency"] = normalized_frequency
        if date:
            reminder_data["date"] = date
        if chat_id:
            reminder_data["chat_id"] = chat_id
        
        result = await self.insert_data(user_id, access_token, reminder_data)
        
        if "successfully created" in result:
            return f"Reminder successfully set{f' for {normalized_frequency}' if normalized_frequency else ''}{f' on {date}' if date else ''}"
        
        return result
    
    async def update_reminder(
        self, 
        user_id: str, 
        access_token: str,
        reminder_id: str,
        frequency: Optional[str] = None,
        date: Optional[str] = None
    ) -> str:
        """Update an existing reminder using enhanced base functionality.
        
        Args:
            user_id: The authenticated user's ID
            access_token: The user's access token
            reminder_id: ID of the reminder to update
            frequency: New reminder frequency
            date: New reminder date
            
        Returns:
            str: Success message or error message
        """
        update_data = {}
        if frequency:
            update_data["frequency"] = frequency
        if date:
            update_data["date"] = date
        
        return await self.update_data(user_id, access_token, reminder_id, update_data)


# Create a singleton instance
_reminder_manager = ReminderManager()


@tool
async def fetch_user_reminders(user_id: str = None, access_token: str = None) -> str:
    """Fetch all reminders for the current authenticated user from the log_reminder table.
    
    This tool retrieves the user's reminder history including frequency, dates,
    and creation timestamps to provide context for conversations.
    
    Args:
        user_id: The authenticated user's ID (will be injected at runtime)
        access_token: The user's access token (will be injected at runtime)
        
    Returns:
        str: JSON string containing the user's reminder data or error message
    """
    return await _reminder_manager.fetch_data(user_id, access_token)


@tool
async def set_user_reminder(
    frequency: str = None,
    date: str = None,
    user_id: str = None, 
    access_token: str = None,
    chat_id: str = None
) -> str:
    """Set a reminder for the current authenticated user in the log_reminder table.
    
    This tool creates reminders for commitments or other important events.
    
    Args:
        frequency: Reminder frequency (daily, weekly, monthly, etc.)
        date: Specific date for the reminder (YYYY-MM-DD format)
        user_id: The authenticated user's ID (will be injected at runtime)
        access_token: The user's access token (will be injected at runtime)
        chat_id: Optional chat session ID for tracking
        
    Returns:
        str: Success message with confirmation or error message
    """
    return await _reminder_manager.set_reminder(
        user_id=user_id,
        access_token=access_token,
        frequency=frequency,
        date=date,
        chat_id=chat_id
    )


@tool
async def update_user_reminder(
    reminder_id: str,
    frequency: str = None,
    date: str = None,
    user_id: str = None, 
    access_token: str = None
) -> str:
    """Update an existing reminder for the current authenticated user.
    
    This tool updates reminder frequency or date for existing reminders.
    
    Args:
        reminder_id: ID of the reminder to update
        frequency: New reminder frequency (daily, weekly, monthly, etc.)
        date: New specific date for the reminder (YYYY-MM-DD format)
        user_id: The authenticated user's ID (will be injected at runtime)
        access_token: The user's access token (will be injected at runtime)
        
    Returns:
        str: Success message with confirmation or error message
    """
    return await _reminder_manager.update_reminder(
        user_id=user_id,
        access_token=access_token,
        reminder_id=reminder_id,
        frequency=frequency,
        date=date
    )
