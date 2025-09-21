"""Enhanced base class for Supabase tools.

This module provides a comprehensive base class that handles common Supabase operations
like authentication, client creation, CRUD operations, and error handling for LangGraph tools.
"""

import json
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from supabase import (
    Client,
    create_client,
)

from app.core.logging import logger
from app.utils.supabase_auth import supabase_auth


class BaseSupabaseTool(ABC):
    """Base class for tools that fetch data from Supabase tables."""
    
    def __init__(self, table_name: str, tool_name: str):
        """Initialize the base tool.
        
        Args:
            table_name: Name of the Supabase table to query
            tool_name: Name of the tool for logging purposes
        """
        self.table_name = table_name
        self.tool_name = tool_name
    
    def _validate_auth_context(self, user_id: str, access_token: str) -> Optional[str]:
        """Validate user authentication context.
        
        Args:
            user_id: The user ID to validate
            access_token: The access token to validate
            
        Returns:
            Error message if validation fails, None if successful
        """
        logger.info(
            f"{self.tool_name}_called",
            has_user_id=bool(user_id),
            has_access_token=bool(access_token),
            user_id=user_id
        )
        
        if not user_id or not access_token:
            logger.error(
                f"{self.tool_name}_missing_context",
                user_id=user_id,
                has_access_token=bool(access_token)
            )
            return "Error: User authentication context not available"
        
        # Token is already verified at the API level by get_current_user dependency
        # No need to re-verify here - just log that we have the required context
        return None
    
    def _create_authenticated_client(self, access_token: str) -> Optional[Client]:
        """Create a Supabase client with user authentication.
        
        Args:
            access_token: The user's access token
            
        Returns:
            Authenticated Supabase client or None if creation fails
        """
        if not supabase_auth.supabase_url or not supabase_auth.supabase_key:
            logger.error("supabase_credentials_not_available")
            return None
        
        try:
            # Create a new client instance with the user's token
            supabase_client = create_client(supabase_auth.supabase_url, supabase_auth.supabase_key)
            
            # Set authorization header for database requests
            supabase_client.postgrest.auth(access_token)
            
            return supabase_client
        except Exception as e:
            logger.error(
                f"{self.tool_name}_client_creation_error",
                error=str(e)
            )
            return None
    
    @abstractmethod
    def _build_query(self, client: Client, user_id: str) -> Any:
        """Build the database query for this specific tool.
        
        Args:
            client: Authenticated Supabase client
            user_id: User ID to query for
            
        Returns:
            Query result from Supabase
        """
        pass
    
    @abstractmethod
    def _format_results(self, data: List[Dict], user_id: str) -> Dict[str, Any]:
        """Format the query results for AI consumption.
        
        Args:
            data: Raw data from Supabase query
            user_id: User ID for context
            
        Returns:
            Formatted data dictionary
        """
        pass
    
    def _get_no_data_message(self) -> str:
        """Get the message to return when no data is found.
        
        Returns:
            No data message for this tool
        """
        return f"No {self.table_name.replace('_', ' ')} found for this user."
    
    async def fetch_data(self, user_id: str = None, access_token: str = None) -> str:
        """Fetch data from Supabase table with proper authentication.
        
        Args:
            user_id: The authenticated user's ID
            access_token: The user's access token
            
        Returns:
            JSON string containing the data or error message
        """
        try:
            # Validate authentication context (basic checks only - token already verified at API level)
            auth_error = self._validate_auth_context(user_id, access_token)
            if auth_error:
                return auth_error
            
            # Create authenticated client
            supabase_client = self._create_authenticated_client(access_token)
            if not supabase_client:
                return "Error: Supabase credentials not properly configured"
            
            # Execute the query
            result = self._build_query(supabase_client, user_id)
            
            if result.data is None:
                logger.info(f"{self.tool_name}_no_data", user_id=user_id)
                return self._get_no_data_message()
            
            data = result.data
            logger.info(
                f"{self.tool_name}_success",
                user_id=user_id,
                count=len(data)
            )
            
            if not data:
                return self._get_no_data_message()
            
            # Format and return results
            formatted_data = self._format_results(data, user_id)
            return json.dumps(formatted_data, indent=2)
            
        except Exception as e:
            error_message = f"Error fetching {self.table_name.replace('_', ' ')}: {str(e)}"
            logger.error(
                f"{self.tool_name}_error",
                user_id=user_id if user_id else 'unknown',
                error=str(e)
            )
            return error_message
    
    async def insert_data(
        self, 
        user_id: str, 
        access_token: str, 
        data: Dict[str, Any],
        custom_validation: Optional[callable] = None
    ) -> str:
        """Insert data into Supabase table with validation.
        
        Args:
            user_id: The authenticated user's ID
            access_token: The user's access token
            data: Data dictionary to insert
            custom_validation: Optional custom validation function
            
        Returns:
            Success message or error message
        """
        try:
            # Validate authentication context
            auth_error = self._validate_auth_context(user_id, access_token)
            if auth_error:
                return auth_error
            
            # Custom validation if provided
            if custom_validation:
                validation_error = await custom_validation(user_id, access_token, data)
                if validation_error:
                    return validation_error
            
            # Create authenticated client
            supabase_client = self._create_authenticated_client(access_token)
            if not supabase_client:
                return "Error: Supabase credentials not properly configured"
            
            # Ensure user_id is set
            data["user_id"] = user_id
            
            # Insert data into database
            result = supabase_client.table(self.table_name).insert(data).execute()
            
            if result.data:
                logger.info(
                    f"{self.tool_name}_insert_success",
                    user_id=user_id,
                    record_id=result.data[0].get("id") if result.data else None
                )
                return self._format_insert_success(result.data[0], data)
            else:
                logger.error(f"{self.tool_name}_no_data_returned", user_id=user_id)
                return f"Error: Failed to create {self.table_name.replace('_', ' ')} - no data returned"
            
        except Exception as e:
            error_message = f"Error creating {self.table_name.replace('_', ' ')}: {str(e)}"
            logger.error(
                f"{self.tool_name}_insert_error",
                user_id=user_id if user_id else 'unknown',
                error=str(e)
            )
            return error_message
    
    async def update_data(
        self, 
        user_id: str, 
        access_token: str, 
        record_id: str,
        update_data: Dict[str, Any]
    ) -> str:
        """Update data in Supabase table.
        
        Args:
            user_id: The authenticated user's ID
            access_token: The user's access token
            record_id: ID of the record to update
            update_data: Data to update
            
        Returns:
            Success message or error message
        """
        try:
            # Validate authentication context
            auth_error = self._validate_auth_context(user_id, access_token)
            if auth_error:
                return auth_error
            
            # Create authenticated client
            supabase_client = self._create_authenticated_client(access_token)
            if not supabase_client:
                return "Error: Supabase credentials not properly configured"
            
            if not update_data:
                return "Error: No update data provided"
            
            # Update record in database
            result = supabase_client.table(self.table_name).update(update_data).eq("id", record_id).eq("user_id", user_id).execute()
            
            if result.data:
                logger.info(
                    f"{self.tool_name}_update_success",
                    user_id=user_id,
                    record_id=record_id
                )
                return f"{self.table_name.replace('_', ' ').title()} successfully updated"
            else:
                logger.warning(f"{self.tool_name}_no_update", user_id=user_id, record_id=record_id)
                return f"No {self.table_name.replace('_', ' ')} found to update or no changes made"
            
        except Exception as e:
            error_message = f"Error updating {self.table_name.replace('_', ' ')}: {str(e)}"
            logger.error(
                f"{self.tool_name}_update_error",
                user_id=user_id if user_id else 'unknown',
                error=str(e)
            )
            return error_message
    
    def _format_insert_success(self, inserted_data: Dict[str, Any], original_data: Dict[str, Any]) -> str:
        """Format success message for insert operations.
        
        Args:
            inserted_data: Data returned from database insert
            original_data: Original data that was inserted
            
        Returns:
            Formatted success message with record ID
        """
        record_id = inserted_data.get("id", "")
        return f"{self.table_name.replace('_', ' ').title()} successfully created [ID: {record_id}]"
