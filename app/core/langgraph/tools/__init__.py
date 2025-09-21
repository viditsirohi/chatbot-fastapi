"""LangGraph tools for enhanced language model capabilities.

This package contains custom tools that can be used with LangGraph to extend
the capabilities of language models. Focuses on user data management and
coaching support functionality.
"""

from langchain_core.tools.base import BaseTool

from .commitment_enhanced import (
    complete_user_commitment,
    create_user_commitment,
    fetch_user_commitments_enhanced,
)
from .journal_fetch import fetch_user_journal_entries
from .reminder_manage import (
    fetch_user_reminders,
    set_user_reminder,
    update_user_reminder,
)
from .reminder_offer import (
    decline_commitment_reminder,
    offer_commitment_reminder,
    set_commitment_reminder,
)

tools: list[BaseTool] = [
    fetch_user_commitments_enhanced,
    create_user_commitment,
    complete_user_commitment,
    fetch_user_journal_entries, 
    fetch_user_reminders,
    set_user_reminder,
    update_user_reminder,
    offer_commitment_reminder,
    set_commitment_reminder,
    decline_commitment_reminder
]
