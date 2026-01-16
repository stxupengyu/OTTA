"""
Data models and constants.
"""

from dataclasses import dataclass
from typing import List


# Default configuration constants
DEFAULT_MODEL = "gpt-3.5-turbo"  # Default model name
DEFAULT_BASE_URL = None  # Use OpenAI default base URL when None
DEFAULT_OPENAI_API_KEY = ""  # Set via environment variable or CLI
DEFAULT_SILICONFLOW_API_KEY = ""  # Set via environment variable or CLI


@dataclass
class Example:
    """Data class for a test example."""
    idx: int  # Example index
    text: str  # Text to classify
    labels: List[str]  # Ground-truth labels
