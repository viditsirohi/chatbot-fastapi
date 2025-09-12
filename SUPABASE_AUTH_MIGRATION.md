# Supabase Authentication Migration

This document outlines the changes made to migrate from custom JWT authentication to Supabase-based authentication.

## Changes Made

### 1. **Removed Custom Authentication Endpoints**
- **Removed endpoints:**
  - `POST /api/v1/auth/register` - User registration
  - `POST /api/v1/auth/login` - User login  
  - `POST /api/v1/auth/session` - Create chat session
  - `PATCH /api/v1/auth/session/{session_id}/name` - Update session name
  - `DELETE /api/v1/auth/session/{session_id}` - Delete session
  - `GET /api/v1/auth/sessions` - Get user sessions

### 2. **New Supabase Authentication System**
- **Created `app/utils/supabase_auth.py`:** New Supabase authentication utility
- **Updated `app/api/v1/auth.py`:** Now contains only authentication dependencies
- **New dependencies:**
  - `get_current_user()` - Returns Supabase user information as dict
  - `get_current_session_id()` - Generates session ID based on user

### 3. **Updated Chatbot Endpoints**
All chatbot endpoints now use Supabase authentication:
- `POST /api/v1/chatbot/chat`
- `POST /api/v1/chatbot/chat/stream`  
- `GET /api/v1/chatbot/messages`
- `DELETE /api/v1/chatbot/messages`

### 4. **Configuration Changes**
- **Added to `app/core/config.py`:**
  - `SUPABASE_URL` - Your Supabase project URL
  - `SUPABASE_ANON_KEY` - Your Supabase anonymous key
- **Removed JWT-related configurations**
- **Removed rate limiting for auth endpoints**

### 5. **Database Models**
- Local User and Session models are no longer used for authentication
- Chat history still uses LangGraph's PostgreSQL checkpointing
- Session IDs are now generated dynamically per request

## Environment Variables Required

Add these to your `.env` files:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-supabase-anon-key
```

## How It Works Now

1. **Frontend Authentication:**
   - User authenticates with Supabase (registration/login handled by frontend)
   - Frontend receives Supabase access token
   - Frontend sends access token in Authorization header

2. **Backend Token Validation:**
   - Each API request validates Supabase token using `get_current_user()`
   - Token validation returns user information (id, email, etc.)
   - Session ID is generated dynamically per user

3. **Chat Sessions:**
   - No longer stored in local database
   - Session IDs generated based on user ID
   - Chat history maintained via LangGraph checkpointing

## API Usage Examples

### Before (Custom Auth):
```bash
# Register user
POST /api/v1/auth/register
{
  "email": "user@example.com",
  "password": "password123"
}

# Create session  
POST /api/v1/auth/session
Authorization: Bearer <custom-jwt-token>
```

### After (Supabase Auth):
```bash
# Chat directly with Supabase token
POST /api/v1/chatbot/chat
Authorization: Bearer <supabase-access-token>
{
  "messages": [{"role": "user", "content": "Hello"}]
}
```

## Benefits

1. **Simplified Architecture:** No custom user management
2. **External Authentication:** Leverage Supabase's auth features
3. **Scalability:** Reduced local database operations
4. **Security:** Professional authentication system
5. **Frontend Integration:** Direct Supabase client usage

## Migration Notes

- **Breaking Change:** All existing JWT tokens are invalid
- **Frontend Update Required:** Must integrate Supabase auth client
- **Database Cleanup:** Local user/session tables can be removed
- **Session Persistence:** Chat history maintained via thread IDs
