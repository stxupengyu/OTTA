"""
Prompt construction helpers for the LLM.
"""

import json
from typing import Dict, List, Optional


def build_llm_prompt(
    text: str,
    label_to_desc: Dict[str, str],
    use_label_desc: bool = True,
    rag_examples: Optional[List[Dict]] = None,
    dataset_desc: Optional[str] = None,
) -> str:
    """
    Build the multi-label classification prompt.

    Args:
        text: Text to classify
        label_to_desc: Mapping from label to description
        use_label_desc: Whether to include label descriptions
        rag_examples: Optional RAG examples with "text" and "prediction" (labels)
        dataset_desc: Optional dataset description for task context

    Returns:
        Prompt string
    """
    # Format labels/label descriptions
    desc_lines = []
    for label, desc in label_to_desc.items():
        if use_label_desc:
            # Collapse whitespace in descriptions
            short_desc = " ".join(desc.split())
            desc_lines.append(f"- {label}: {short_desc}")
        else:
            # Only use label names
            desc_lines.append(f"- {label}")

    labels_str = "\n".join(desc_lines)

    # Example JSON output
    example_output = {
        "labels": list(label_to_desc.keys())[:3]  # Example: take first 3 labels
    }

    # Dataset description section
    dataset_desc_section = ""
    if dataset_desc:
        dataset_desc_section = f"""
Task description:
{dataset_desc}

"""

    # RAG examples section
    rag_section = ""
    if rag_examples and len(rag_examples) > 0:
        rag_examples_str = []
        for i, rag_ex in enumerate(rag_examples, 1):
            rag_text = rag_ex.get("text", "")
            rag_pred = rag_ex.get("prediction", {})
            rag_labels = rag_pred.get("labels", []) if isinstance(rag_pred, dict) else []
            rag_examples_str.append(
                f"Example {i}:\n"
                f"Text: {rag_text}\n"
                f"Predicted labels: {', '.join(rag_labels)}"
            )
        rag_section = f"""

Reference examples (high-confidence predictions from earlier steps):
{chr(10).join(rag_examples_str)}

"""

    prompt = f"""
You are a multi-label classifier. Select all applicable labels from the list below and output JSON:
{dataset_desc_section}
{labels_str}
{rag_section}
Output requirements:
1. Output JSON only, no extra text.
2. JSON must include the "labels" field (string array).
3. Select all applicable labels, up to 5 total.

Example output:
{json.dumps(example_output, ensure_ascii=False, indent=2)}

Text to classify:
{text}
"""
    return prompt.strip()
