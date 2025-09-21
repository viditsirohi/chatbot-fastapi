"""This file contains the prompts for the agent."""

import os
from datetime import datetime

from app.core.config import settings


def load_knowledge_file(filename: str) -> str:
    """Load a knowledge file from the knowledge directory.
    
    Args:
        filename: Name of the knowledge file (with extension)
        
    Returns:
        str: The loaded knowledge content
    """
    knowledge_dir = os.path.join(os.path.dirname(__file__), "..", "knowledge")
    filepath = os.path.join(knowledge_dir, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Knowledge file {filename} not found."


def load_prompt_file(filename: str) -> str:
    """Load a prompt from a markdown file.
    
    Args:
        filename: Name of the prompt file (without .md extension)
        
    Returns:
        str: The loaded prompt template (unformatted for brain, formatted for others)
    """
    filepath = os.path.join(os.path.dirname(__file__), f"{filename}.md")
    with open(filepath, "r") as f:
        content = f.read()
        
        # For brain prompt, return template for later formatting with knowledge
        if filename == "brain":
            return content
        
        # For other prompts, format immediately
        return content.format(
            agent_name=settings.PROJECT_NAME + " Agent",
            current_date_and_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )


def load_system_prompt():
    """Load the system prompt from the file."""
    return load_prompt_file("system")


def load_brain_prompt():
    """Load the brain node prompt from the file with knowledge integration."""
    base_prompt = load_prompt_file("brain")
    
    # Load knowledge files
    archetypes_content = load_knowledge_file("archetypes.txt")
    model_principles_content = load_knowledge_file("model_principles.txt")
    
    # Format the prompt with all variables
    return base_prompt.format(
        agent_name=settings.PROJECT_NAME + " Agent",
        current_date_and_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        archetypes_knowledge=archetypes_content,
        model_principles_knowledge=model_principles_content
    )


def load_synthesizer_prompt():
    """Load the synthesizer node prompt from the file."""
    return load_prompt_file("synthesizer")


# Load all prompts
SYSTEM_PROMPT = load_system_prompt()
BRAIN_PROMPT = load_brain_prompt()
SYNTHESIZER_PROMPT = load_synthesizer_prompt()
