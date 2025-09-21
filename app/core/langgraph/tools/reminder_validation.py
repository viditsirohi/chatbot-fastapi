"""Reminder validation utilities.

This module provides validation functions for reminder data to ensure
compatibility with the database schema and enum constraints.
"""

from typing import (
    Optional,
    Set,
)

# Define allowed frequency values based on the database enum
# Note: These match the public.reminder_frequency enum in PostgreSQL exactly
ALLOWED_FREQUENCIES: Set[str] = {
    "daily",
    "weekly", 
    "fortnightly",
    "monthly"
}


def validate_reminder_frequency(frequency: Optional[str]) -> Optional[str]:
    """Validate that frequency value is allowed by database enum.
    
    Args:
        frequency: The frequency value to validate
        
    Returns:
        Error message if invalid, None if valid
    """
    if frequency is None:
        return None
    
    if frequency.lower() not in ALLOWED_FREQUENCIES:
        allowed_list = ", ".join(sorted(ALLOWED_FREQUENCIES))
        return f"Invalid frequency '{frequency}'. Allowed values: {allowed_list}"
    
    return None


def validate_reminder_date(date: Optional[str]) -> Optional[str]:
    """Validate that date is in correct format for database.
    
    Args:
        date: The date value to validate (YYYY-MM-DD format)
        
    Returns:
        Error message if invalid, None if valid
    """
    if date is None:
        return None
    
    import re
    from datetime import datetime

    # Check basic format
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
        return f"Invalid date format '{date}'. Use YYYY-MM-DD format"
    
    # Check if date is valid
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        return f"Invalid date '{date}'. Please provide a valid date"
    
    # Check if date is not in the past
    from datetime import date as date_obj
    try:
        reminder_date = datetime.strptime(date, '%Y-%m-%d').date()
        today = date_obj.today()
        
        if reminder_date < today:
            return f"Date '{date}' is in the past. Please choose a future date"
    except Exception:
        pass  # If comparison fails, let database handle it
    
    return None


def validate_reminder_data(frequency: Optional[str], date: Optional[str]) -> Optional[str]:
    """Validate complete reminder data for database compatibility.
    
    Args:
        frequency: The frequency value
        date: The date value
        
    Returns:
        Error message if invalid, None if valid
    """
    # Validate frequency
    freq_error = validate_reminder_frequency(frequency)
    if freq_error:
        return freq_error
    
    # Validate date
    date_error = validate_reminder_date(date)
    if date_error:
        return date_error
    
    # Validate logic - must have either frequency or date, but not both
    has_frequency = frequency is not None
    has_date = date is not None
    
    if not has_frequency and not has_date:
        return "Must specify either frequency or date for reminder"
    
    if has_frequency and has_date:
        return "Cannot specify both frequency and date. Choose one option only"
    
    return None


def normalize_frequency(frequency: Optional[str]) -> Optional[str]:
    """Normalize frequency value to match database enum.
    
    Args:
        frequency: The frequency value to normalize
        
    Returns:
        Normalized frequency value or None
    """
    if frequency is None:
        return None
    
    normalized = frequency.lower().strip()
    
    # Handle common variations
    if normalized in ["daily", "day", "everyday", "every day"]:
        return "daily"
    elif normalized in ["weekly", "week", "every week"]:
        return "weekly"
    elif normalized in ["fortnightly", "fortnight", "every fortnight", "every two weeks", "bi-weekly", "biweekly"]:
        return "fortnightly"
    elif normalized in ["monthly", "month", "every month"]:
        return "monthly"
    
    return normalized if normalized in ALLOWED_FREQUENCIES else frequency
