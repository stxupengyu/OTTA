"""
Data loading helpers.
"""

import csv
from typing import Dict, List

from models import Example


def load_tag_descriptions(path: str) -> Dict[str, str]:
    """
    Load label descriptions from a CSV file.

    Args:
        path: CSV file path with "tags" and "outputs" columns

    Returns:
        Dict mapping tag -> description
    """
    label_to_desc: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = (row.get("tags") or "").strip()
            if not tag:
                continue
            desc = (row.get("outputs") or "").strip()
            # Normalize line endings
            label_to_desc[tag] = desc.replace("\r\n", "\n").strip()
    return label_to_desc


def load_test_file(path: str) -> List[Example]:
    """
    Load test samples from a file.

    File format:
        Text: <text>
        Labels: <label1>,<label2>,...
        (empty lines separate samples)

    Args:
        path: Test file path

    Returns:
        List of Example objects
    """
    examples: List[Example] = []
    current_text = None  # Current text being read
    idx = 0  # Example counter

    def flush(labels_line: str) -> None:
        """Create an example from current text + labels."""
        nonlocal idx, current_text
        if current_text is None:
            return
        # Parse comma-separated labels
        labels = [lab.strip() for lab in labels_line.split(",") if lab.strip()]
        idx += 1
        examples.append(Example(idx=idx, text=current_text.strip(), labels=labels))
        current_text = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse "Text:" line
            if line.startswith("Text:"):
                current_text = line[len("Text:"):].strip()
            # Parse "Labels:" line and flush the sample
            elif line.startswith("Labels:"):
                flush(line[len("Labels:"):].strip())

    return examples
