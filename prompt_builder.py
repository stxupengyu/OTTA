"""
LLM prompt builder.
"""

import json
from typing import Dict, List, Optional


def build_llm_prompt(
    text: str,
    label_to_desc: Dict[str, str],
    use_label_desc: bool = True,
    rag_examples: Optional[List[Dict]] = None,
) -> str:
    """
    Build a multi-label classification prompt for the LLM.

    Args:
        text: Input text to classify
        label_to_desc: Label -> description mapping
        use_label_desc: Whether to include label descriptions
        rag_examples: Optional RAG examples, each with "text" and "prediction" (labels)

    Returns:
        Prompt string
    """
    # Format label list
    desc_lines = []
    for label, desc in label_to_desc.items():
        if use_label_desc:
            # Collapse whitespace
            short_desc = " ".join(desc.split())
            desc_lines.append(f"- {label}: {short_desc}")
        else:
            # Use label name only
            desc_lines.append(f"- {label}")
    
    labels_str = "\n".join(desc_lines)
    
    # JSON output example
    example_output = {
        "labels": list(label_to_desc.keys())[:3]  # Example: first 3 labels
    }
    
    # Build RAG example section
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

Reference examples (previous high-confidence predictions):
{chr(10).join(rag_examples_str)}

"""
    
    prompt = f"""
You are a multi-label classifier. Select all applicable labels from the list and output JSON:

{labels_str}
{rag_section}
Output requirements:
1. Output JSON only, no extra text.
2. JSON must include a "labels" field (string array).
3. Select all labels you consider applicable.
4. Do not output a "confidences" field.

Example output:
{json.dumps(example_output, ensure_ascii=False, indent=2)}

Text to classify:
{text}
"""
    return prompt.strip()
