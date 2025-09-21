"""Supabase authentication utilities for the application."""

import os
import uuid
from typing import Optional

from fastapi import HTTPException
from supabase import (
    Client,
    create_client,
)

from app.core.config import settings
from app.core.logging import logger


class SupabaseAuth:
    """Supabase authentication service."""
    
    def __init__(self):
        """Initialize Supabase client."""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            logger.warning("supabase_credentials_missing", supabase_url=bool(self.supabase_url), supabase_key=bool(self.supabase_key))
            # Don't raise error during initialization, handle it when methods are called
            self.supabase: Optional[Client] = None
            return
        
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info("supabase_client_initialized", url=self.supabase_url)
        except Exception as e:
            logger.error("supabase_client_initialization_failed", error=str(e))
            self.supabase = None

    async def verify_token_and_get_user(self, access_token: str) -> dict:
        """Verify Supabase access token and return user information.
        
        Args:
            access_token: The Supabase access token to verify
            
        Returns:
            dict: User information from Supabase
            
        Raises:
            HTTPException: If token is invalid or user not found
        """
        if not self.supabase:
            logger.error("supabase_client_not_initialized")
            raise HTTPException(
                status_code=500,
                detail="Supabase client not properly configured",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        try:
            # Use the token directly - JWT tokens should not be HTML sanitized
            token = access_token.strip()
            
            # Get user from Supabase using the JWT token directly
            user_response = self.supabase.auth.get_user(token)
            
            if not user_response.user:
                logger.error("supabase_user_not_found", token_part=token[:10] + "...")
                raise HTTPException(
                    status_code=401,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            user_data = {
                "id": user_response.user.id,
                "email": user_response.user.email,
                "created_at": user_response.user.created_at,
                "last_sign_in_at": user_response.user.last_sign_in_at,
            }
            
            logger.info("supabase_user_verified", user_id=user_data["id"], email=user_data["email"])
            return user_data
            
        except HTTPException:
            # Re-raise HTTPExceptions as-is
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error("supabase_token_verification_failed", error=error_msg, token_part=access_token[:10] + "...")
            
            # Provide more specific error messages based on the error type
            if "Auth session missing" in error_msg:
                detail = "Authentication session expired or invalid"
            elif "Invalid JWT" in error_msg or "invalid token" in error_msg.lower():
                detail = "Invalid authentication token format"
            elif "expired" in error_msg.lower():
                detail = "Authentication token has expired"
            else:
                detail = "Invalid authentication credentials"
                
            raise HTTPException(
                status_code=401,
                detail=detail,
                headers={"WWW-Authenticate": "Bearer"},
            )


# Create a singleton instance  
supabase_auth = SupabaseAuth()
