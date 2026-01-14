"""
Evaluation metric helpers.
"""

from typing import Dict, List, Sequence, Set, Tuple


def compute_micro_f1(
    predictions: Sequence[Sequence[str]],
    gold_labels: Sequence[Sequence[str]],
) -> float:
    """
    Compute micro-F1.

    Micro-F1 is the harmonic mean of global precision and recall.

    Args:
        predictions: Predicted labels per example
        gold_labels: Gold labels per example

    Returns:
        Micro-F1 score in [0, 1]
    """
    all_tp = 0  # True Positives
    all_fp = 0  # False Positives
    all_fn = 0  # False Negatives
    
    for pred, gold in zip(predictions, gold_labels):
        pred_set = set(pred)
        gold_set = set(gold)
        
        tp = len(pred_set.intersection(gold_set))  # True positives
        fp = len(pred_set - gold_set)  # False positives
        fn = len(gold_set - pred_set)  # False negatives
        
        all_tp += tp
        all_fp += fp
        all_fn += fn
    
    # Global precision and recall
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    
    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1


def compute_macro_f1(
    predictions: Sequence[Sequence[str]],
    gold_labels: Sequence[Sequence[str]],
    all_labels: Sequence[str],
) -> float:
    """
    Compute macro-F1.

    Macro-F1 is the average F1 over labels.

    Args:
        predictions: Predicted labels per example
        gold_labels: Gold labels per example
        all_labels: Full label space

    Returns:
        Macro-F1 score in [0, 1]
    """
    label_f1s = []
    
    for label in all_labels:
        tp = 0  # True positives for this label
        fp = 0  # False positives for this label
        fn = 0  # False negatives for this label
        
        for pred, gold in zip(predictions, gold_labels):
            pred_set = set(pred)
            gold_set = set(gold)
            
            if label in pred_set and label in gold_set:
                tp += 1
            elif label in pred_set and label not in gold_set:
                fp += 1
            elif label not in pred_set and label in gold_set:
                fn += 1
        
        # Precision and recall for this label
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 for this label
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        label_f1s.append(f1)
    
    # Average across labels
    macro_f1 = sum(label_f1s) / len(label_f1s) if label_f1s else 0.0
    return macro_f1


def compute_example_f1(
    predictions: Sequence[Sequence[str]],
    gold_labels: Sequence[Sequence[str]],
) -> float:
    """
    Compute example-F1 (a.k.a. sample-F1).

    Example-F1 is the average F1 across examples.

    Args:
        predictions: Predicted labels per example
        gold_labels: Gold labels per example

    Returns:
        Example-F1 score in [0, 1]
    """
    example_f1s = []
    
    for pred, gold in zip(predictions, gold_labels):
        pred_set = set(pred)
        gold_set = set(gold)
        
        if not pred_set and not gold_set:
            # Both empty: perfect match
            f1 = 1.0
        elif not pred_set or not gold_set:
            # One empty: no overlap
            f1 = 0.0
        else:
            tp = len(pred_set.intersection(gold_set))
            precision = tp / len(pred_set) if pred_set else 0.0
            recall = tp / len(gold_set) if gold_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        example_f1s.append(f1)
    
    # Average across examples
    example_f1 = sum(example_f1s) / len(example_f1s) if example_f1s else 0.0
    return example_f1


def compute_metrics(
    predictions: Sequence[Sequence[str]],
    gold_labels: Sequence[Sequence[str]],
    all_labels: Sequence[str],
) -> Dict[str, float]:
    """
    Compute all metrics (micro-f1, macro-f1, example-f1).

    Args:
        predictions: Predicted labels per example
        gold_labels: Gold labels per example
        all_labels: Full label space

    Returns:
        Dict of metric name -> score (percent)
    """
    micro_f1 = compute_micro_f1(predictions, gold_labels)
    macro_f1 = compute_macro_f1(predictions, gold_labels, all_labels)
    example_f1 = compute_example_f1(predictions, gold_labels)
    
    return {
        "micro-f1": round(micro_f1 * 100, 2),
        "macro-f1": round(macro_f1 * 100, 2),
        "example-f1": round(example_f1 * 100, 2),
    }
