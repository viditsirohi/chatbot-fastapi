"""Journal entry fetch tool for LangGraph.

This module provides a tool that fetches journal entries from the Supabase
journal_entry table using the user's access token and user_id. This allows
the AI to access and analyze the user's journal entries.
"""

from typing import (
    Any,
    Dict,
    List,
)

from langchain_core.tools import tool
from supabase import Client

from .base_supabase_tool import BaseSupabaseTool


class JournalFetcher(BaseSupabaseTool):
    """Tool for fetching user journal entries from the journal_entry table."""
    
    def __init__(self):
        super().__init__(table_name="journal_entry", tool_name="journal_tool")
    
    def _build_query(self, client: Client, user_id: str) -> Any:
        """Build query to fetch journal entries for the user, ordered by creation date (newest first)."""
        return client.table("journal_entry").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    
    def _format_results(self, data: List[Dict], user_id: str) -> Dict[str, Any]:
        """Format journal entries data for AI consumption."""
        formatted_entries = []
        for entry in data:
            formatted_entries.append({
                "id": entry.get("id"),
                "journal_entry": entry.get("journal_entry"),
                "user_id": entry.get("user_id"),
                "created_at": entry.get("created_at")
            })
        
        return {
            "user_id": user_id,
            "total_entries": len(formatted_entries),
            "journal_entries": formatted_entries
        }


# Create a singleton instance
_journal_fetcher = JournalFetcher()


@tool
async def fetch_user_journal_entries(user_id: str = None, access_token: str = None) -> str:
    """Fetch all journal entries for the current authenticated user from the journal_entry table.
    
    This tool automatically retrieves the user's journal entries including entry text
    and creation timestamps to provide context for conversations.
    
    Args:
        user_id: The authenticated user's ID (will be injected at runtime)
        access_token: The user's access token (will be injected at runtime)
        
    Returns:
        str: JSON string containing the user's journal entry data or error message
    """
    return await _journal_fetcher.fetch_data(user_id, access_token)
