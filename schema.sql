-- Database schema for the application
-- Note: After Supabase migration, only LangGraph-related tables are needed

-- Create thread table (used by LangGraph for chat history)
CREATE TABLE IF NOT EXISTS thread (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
