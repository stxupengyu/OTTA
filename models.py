"""
Data models and constants.
"""

import os
from dataclasses import dataclass
from typing import List


# Default configuration constants
DEFAULT_MODEL = "gpt-3.5-turbo"  # Default model name
DEFAULT_BASE_URL = None  # None uses the OpenAI default base URL
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")


@dataclass
class Example:
    """Represents a single test example."""
    idx: int  # Example index
    text: str  # Text to classify
    labels: List[str]  # Ground-truth labels
