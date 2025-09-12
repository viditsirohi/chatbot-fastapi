"""This file contains the prompts for the agent."""

import os
from datetime import datetime

from app.core.config import settings


def load_prompt_file(filename: str) -> str:
    """Load a prompt from a markdown file.
    
    Args:
        filename: Name of the prompt file (without .md extension)
        
    Returns:
        str: The loaded and formatted prompt
    """
    filepath = os.path.join(os.path.dirname(__file__), f"{filename}.md")
    with open(filepath, "r") as f:
        return f.read().format(
            agent_name=settings.PROJECT_NAME + " Agent",
            current_date_and_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )


def load_system_prompt():
    """Load the system prompt from the file."""
    return load_prompt_file("system")


def load_brain_prompt():
    """Load the brain node prompt from the file."""
    return load_prompt_file("brain")


def load_synthesizer_prompt():
    """Load the synthesizer node prompt from the file."""
    return load_prompt_file("synthesizer")


# Load all prompts
SYSTEM_PROMPT = load_system_prompt()
BRAIN_PROMPT = load_brain_prompt()
SYNTHESIZER_PROMPT = load_synthesizer_prompt()
