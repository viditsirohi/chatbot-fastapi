"""Enhanced commitment management tools for LangGraph.

This module provides comprehensive commitment management including:
- Fetching user commitments with filtering
- Setting new commitments with 5-limit validation
- Marking commitments as complete
- Rich context tracking for better user experience
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
from app.utils.error_handling import (
    get_user_friendly_error_message,
    is_user_facing_error,
)

from .base_supabase_tool import BaseSupabaseTool


class EnhancedCommitmentManager(BaseSupabaseTool):
    """Enhanced tool for managing user commitments with validation and context."""
    
    def __init__(self):
        super().__init__(table_name="log_commitment", tool_name="enhanced_commitment_manager")
    
    def _build_query(self, client: Client, user_id: str) -> Any:
        """Build query to fetch commitments for the user."""
        return client.table("log_commitment").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    
    def _format_results(self, data: List[Dict], user_id: str) -> Dict[str, Any]:
        """Format commitments data for AI consumption with detailed context."""
        active_commitments = []
        completed_commitments = []
        
        for commitment in data:
            formatted_commitment = {
                "id": commitment.get("id"),
                "commitment": commitment.get("commitment"),
                "chat_id": commitment.get("chat_id"),
                "done": commitment.get("done", False),
                "reminder_set": commitment.get("reminder_set", False),
                "created_at": commitment.get("created_at")
            }
            
            if commitment.get("done", False):
                completed_commitments.append(formatted_commitment)
            else:
                active_commitments.append(formatted_commitment)
        
        return {
            "user_id": user_id,
            "total_commitments": len(data),
            "active_commitments": active_commitments,
            "active_count": len(active_commitments),
            "completed_commitments": completed_commitments,
            "completed_count": len(completed_commitments),
            "can_add_more": len(active_commitments) < 5,
            "remaining_slots": max(0, 5 - len(active_commitments))
        }
    
    async def _validate_commitment_limit(self, user_id: str, access_token: str, data: Dict[str, Any]) -> Optional[str]:
        """Validate that user doesn't exceed 5 active commitments.
        
        Args:
            user_id: The authenticated user's ID
            access_token: The user's access token
            data: Commitment data being added
            
        Returns:
            Error message if validation fails, None if successful
        """
        try:
            # Create authenticated client
            supabase_client = self._create_authenticated_client(access_token)
            if not supabase_client:
                return get_user_friendly_error_message(
                    "Supabase credentials not properly configured",
                    context=self.tool_name,
                    user_id=user_id
                )
            
            # Count active commitments
            result = supabase_client.table("log_commitment").select("id").eq("user_id", user_id).eq("done", False).execute()
            
            active_count = len(result.data) if result.data else 0
            
            if active_count >= 5:
                return f"You already have {active_count} active commitments. Please complete some existing commitments before adding new ones. I can help you work on completing your current commitments if you'd like."
            
            logger.info(
                f"{self.tool_name}_validation_passed",
                user_id=user_id,
                active_count=active_count,
                remaining_slots=5 - active_count
            )
            return None
            
        except Exception as e:
            logger.error(
                f"{self.tool_name}_validation_error",
                user_id=user_id,
                error=str(e)
            )
            # Check if this is a user-facing error message
            if is_user_facing_error(str(e)):
                return str(e)
            else:
                return get_user_friendly_error_message(
                    error=e,
                    context=self.tool_name,
                    user_id=user_id
                )
    
    async def create_commitment(
        self, 
        user_id: str, 
        access_token: str, 
        commitment_text: str,
        chat_id: Optional[str] = None,
        reminder_set: bool = False
    ) -> str:
        """Create a new commitment with proper validation.
        
        Args:
            user_id: The authenticated user's ID
            access_token: The user's access token
            commitment_text: The commitment text to store
            chat_id: Optional chat session ID
            reminder_set: Whether reminder is set for this commitment
            
        Returns:
            Success message or error message
        """
        commitment_data = {
            "commitment": commitment_text,
            "done": False,
            "reminder_set": reminder_set
        }
        
        if chat_id:
            commitment_data["chat_id"] = chat_id
        
        result = await self.insert_data(
            user_id=user_id,
            access_token=access_token,
            data=commitment_data,
            custom_validation=self._validate_commitment_limit
        )
        
        if "successfully created" in result:
            # Extract commitment ID from the database response for reminder offering
            import re
            id_match = re.search(r'\[ID: (.+?)\]', result)
            commitment_id = id_match.group(1) if id_match else ""
            
            success_msg = f"✅ Commitment successfully set: '{commitment_text}'\n\n"
            success_msg += "Your commitment has been saved and you can track it on your home screen.\n\n"
            success_msg += f"[OFFER_REMINDER: true, COMMITMENT_ID: {commitment_id}]"  # Signal to brain to offer reminder
            return success_msg
        
        return result
    
    async def complete_commitment(
        self, 
        user_id: str, 
        access_token: str, 
        commitment_id: str
    ) -> str:
        """Mark a commitment as completed.
        
        Args:
            user_id: The authenticated user's ID
            access_token: The user's access token
            commitment_id: ID of the commitment to complete
            
        Returns:
            Success message or error message
        """
        update_data = {
            "done": True
        }
        
        result = await self.update_data(
            user_id=user_id,
            access_token=access_token,
            record_id=commitment_id,
            update_data=update_data
        )
        
        if "successfully updated" in result:
            return f"🎉 Congratulations! Your commitment has been marked as complete. Great job following through!"
        
        return result
    
    def _format_insert_success(self, inserted_data: Dict[str, Any], original_data: Dict[str, Any]) -> str:
        """Format success message for commitment creation."""
        commitment_text = original_data.get("commitment", "")
        return f"Commitment successfully created: '{commitment_text}'"


# Create a singleton instance
_enhanced_commitment_manager = EnhancedCommitmentManager()


@tool
async def fetch_user_commitments_enhanced(user_id: str = None, access_token: str = None) -> str:
    """Fetch all commitments for the current authenticated user with detailed status information.
    
    This tool retrieves the user's commitment history including active and completed commitments,
    context information, and validates if they can add more commitments (5 active limit).
    
    Args:
        user_id: The authenticated user's ID (will be injected at runtime)
        access_token: The user's access token (will be injected at runtime)
        
    Returns:
        str: JSON string containing detailed commitment data including active/completed counts
    """
    return await _enhanced_commitment_manager.fetch_data(user_id, access_token)


@tool
async def create_user_commitment(
    commitment_text: str,
    user_id: str = None, 
    access_token: str = None,
    chat_id: str = None,
    reminder_set: bool = False
) -> str:
    """Create a new commitment for the current authenticated user with validation.
    
    This tool creates a formal commitment after the user has gone through the complete
    coaching flow. It validates the 5-commitment limit and stores the commitment properly.
    
    Args:
        commitment_text: The specific commitment text the user wants to set
        user_id: The authenticated user's ID (will be injected at runtime)
        access_token: The user's access token (will be injected at runtime)
        chat_id: Optional chat session ID for tracking
        reminder_set: Whether to mark that a reminder is set (default: False)
        
    Returns:
        str: Success message with confirmation or error message (including limit violations)
    """
    # Validate required parameters
    if not commitment_text or not commitment_text.strip():
        return "Please provide the commitment text you'd like to set."
    
    return await _enhanced_commitment_manager.create_commitment(
        user_id=user_id,
        access_token=access_token,
        commitment_text=commitment_text.strip(),
        chat_id=chat_id,
        reminder_set=reminder_set
    )


@tool
async def complete_user_commitment(
    commitment_id: str,
    user_id: str = None, 
    access_token: str = None
) -> str:
    """Mark a commitment as completed for the current authenticated user.
    
    This tool marks an existing commitment as done and records the completion timestamp.
    
    Args:
        commitment_id: ID of the commitment to mark as complete
        user_id: The authenticated user's ID (will be injected at runtime)
        access_token: The user's access token (will be injected at runtime)
        
    Returns:
        str: Success message with celebration or error message
    """
    # Validate required parameters  
    if not commitment_id or not commitment_id.strip():
        return get_user_friendly_error_message(
            "Commitment ID is required and cannot be empty",
            context="complete_user_commitment",
            user_id=user_id
        )
    
    return await _enhanced_commitment_manager.complete_commitment(
        user_id=user_id,
        access_token=access_token,
        commitment_id=commitment_id.strip()
    )
