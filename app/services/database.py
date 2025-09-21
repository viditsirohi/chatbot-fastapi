"""This file contains the database service for the application."""

from datetime import datetime
from typing import (
    Any,
    Dict,
    Optional,
)
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
from sqlmodel import (
    Session,
    create_engine,
    select,
)

from app.core.config import (
    Environment,
    settings,
)
from app.core.logging import logger
from app.schemas.chat import (
    LogChatCreate,
    LogChatResponse,
)
from app.utils.supabase_auth import supabase_auth


class DatabaseService:
    """Service class for database operations.

    This class provides database connectivity and health checks.
    It uses SQLModel for ORM operations and maintains a connection pool.
    """

    def __init__(self):
        """Initialize database service with connection pool."""
        try:
            # Configure environment-specific database connection pool settings
            pool_size = settings.POSTGRES_POOL_SIZE
            max_overflow = settings.POSTGRES_MAX_OVERFLOW

            # Create engine with appropriate pool configuration
            self.engine = create_engine(
                settings.POSTGRES_URL,
                pool_pre_ping=True,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=30,  # Connection timeout (seconds)
                pool_recycle=1800,  # Recycle connections after 30 minutes
            )

            # Database connection established
            # Note: No tables are created automatically - this is for health checks only

            logger.info(
                "database_initialized",
                environment=settings.ENVIRONMENT.value,
                pool_size=pool_size,
                max_overflow=max_overflow,
            )
        except SQLAlchemyError as e:
            logger.error("database_initialization_error", error=str(e), environment=settings.ENVIRONMENT.value)
            # In production, don't raise - allow app to start even with DB issues
            if settings.ENVIRONMENT != Environment.PRODUCTION:
                raise


    def get_session_maker(self):
        """Get a session maker for creating database sessions.

        Returns:
            Session: A SQLModel session maker
        """
        return Session(self.engine)

    async def health_check(self) -> bool:
        """Check database connection health.

        Returns:
            bool: True if database is healthy, False otherwise
        """
        try:
            with Session(self.engine) as session:
                # Execute a simple query to check connection
                session.exec(select(1)).first()
                return True
        except Exception as e:
            logger.error("database_health_check_failed", error=str(e))
            return False

    async def log_chat_interaction(
        self,
        session_id: str,
        user_id: UUID,
        chat_data: Dict[str, Any],
        summary: Optional[str] = None,
        access_token: Optional[str] = None
    ) -> Optional[LogChatResponse]:
        """Log a chat interaction to the log_chat table using Supabase with upsert.
        
        Uses the session_id as the primary key. If the session exists, updates the chat field.
        If it doesn't exist, creates a new record.

        Args:
            session_id: The session ID that becomes the primary key for the chat log
            user_id: The ID of the user who participated in the chat
            chat_data: The chat data as JSON (includes messages and metadata)
            summary: Optional summary of the chat session
            access_token: User's access token for authenticated Supabase operations

        Returns:
            LogChatResponse if successful, None if failed

        Raises:
            Exception: If logging fails and should be handled by caller
        """
        try:
            # Use the existing Supabase client from supabase_auth
            if not supabase_auth.supabase:
                logger.error("log_chat_supabase_client_not_available")
                return None

            # Create log data
            log_data = LogChatCreate(
                id=session_id,
                user_id=user_id,
                chat=chat_data,
                summary=summary
            )

            # Convert to dict for Supabase upsert
            upsert_data = {
                "id": log_data.id,
                "user_id": str(log_data.user_id),
                "chat": log_data.chat,
                "summary": log_data.summary
            }

            # Create authenticated client if access_token is provided
            supabase_client = supabase_auth.supabase
            if access_token:
                try:
                    # Set authorization header for this request
                    supabase_client.postgrest.auth(access_token)
                except Exception as e:
                    logger.warning("log_chat_auth_setup_failed", error=str(e))
                    # Continue with unauthenticated request

            # Upsert into log_chat table using session_id as primary key
            # This will insert if the session doesn't exist, or update if it does
            result = supabase_client.table("log_chat").upsert(
                upsert_data,
                on_conflict="id"  # Use id (session_id) as the conflict resolution key
            ).execute()

            if result.data and len(result.data) > 0:
                log_entry = result.data[0]
                logger.info(
                    "chat_interaction_logged",
                    user_id=user_id,
                    session_id=session_id,
                    log_id=log_entry.get("id"),
                    message_count=len(chat_data) if isinstance(chat_data, list) else 0,
                    operation="upsert"
                )

                # Return structured response
                return LogChatResponse(
                    id=log_entry["id"],
                    created_at=datetime.fromisoformat(log_entry["created_at"].replace('Z', '+00:00')),
                    user_id=UUID(log_entry["user_id"]),
                    chat=log_entry["chat"],
                    summary=log_entry.get("summary")
                )
            else:
                logger.error("log_chat_no_data_returned", user_id=user_id, session_id=session_id)
                return None

        except Exception as e:
            logger.error(
                "log_chat_interaction_failed",
                user_id=user_id,
                session_id=session_id,
                error=str(e),
                exc_info=True
            )
            # Don't re-raise to avoid breaking the main chat flow
            return None


# Create a singleton instance
database_service = DatabaseService()
