"""Supabase authentication dependencies for the API.

This module provides authentication dependencies that validate Supabase tokens
and return user information for protected endpoints.
"""

from fastapi import (
    Depends,
    HTTPException,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from app.core.logging import logger
from app.utils.supabase_auth import supabase_auth

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Get the current user information from Supabase token.

    Args:
        credentials: The HTTP authorization credentials containing the Supabase access token.

    Returns:
        dict: The user information from Supabase.

    Raises:
        HTTPException: If the token is invalid or missing.
    """
    return await supabase_auth.verify_token_and_get_user(credentials.credentials)




