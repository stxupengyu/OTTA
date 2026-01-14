"""
评估指标计算相关函数。
"""

from typing import Dict, List, Sequence, Set, Tuple


def compute_micro_f1(
    predictions: Sequence[Sequence[str]],
    gold_labels: Sequence[Sequence[str]],
) -> float:
    """
    计算 Micro-F1 分数。
    
    Micro-F1: 全局的精确率和召回率的调和平均数。
    先计算所有样本的 TP, FP, FN，然后计算全局的 precision 和 recall，最后计算 F1。
    
    Args:
        predictions: 预测标签列表的列表（每个样本的预测标签）
        gold_labels: 真实标签列表的列表（每个样本的真实标签）
        
    Returns:
        Micro-F1 分数（0-1之间）
    """
    all_tp = 0  # True Positives
    all_fp = 0  # False Positives
    all_fn = 0  # False Negatives
    
    for pred, gold in zip(predictions, gold_labels):
        pred_set = set(pred)
        gold_set = set(gold)
        
        tp = len(pred_set.intersection(gold_set))  # 正确预测的标签数
        fp = len(pred_set - gold_set)  # 错误预测的标签数
        fn = len(gold_set - pred_set)  # 漏掉的标签数
        
        all_tp += tp
        all_fp += fp
        all_fn += fn
    
    # 计算全局精确率和召回率
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    
    # 计算 F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1


def compute_macro_f1(
    predictions: Sequence[Sequence[str]],
    gold_labels: Sequence[Sequence[str]],
    all_labels: Sequence[str],
) -> float:
    """
    计算 Macro-F1 分数。
    
    Macro-F1: 每个标签的 F1 分数的平均值。
    
    Args:
        predictions: 预测标签列表的列表（每个样本的预测标签）
        gold_labels: 真实标签列表的列表（每个样本的真实标签）
        all_labels: 所有可能的标签列表
        
    Returns:
        Macro-F1 分数（0-1之间）
    """
    label_f1s = []
    
    for label in all_labels:
        tp = 0  # True Positives: 该标签被正确预测的次数
        fp = 0  # False Positives: 该标签被错误预测的次数
        fn = 0  # False Negatives: 该标签被漏掉的次数
        
        for pred, gold in zip(predictions, gold_labels):
            pred_set = set(pred)
            gold_set = set(gold)
            
            if label in pred_set and label in gold_set:
                tp += 1
            elif label in pred_set and label not in gold_set:
                fp += 1
            elif label not in pred_set and label in gold_set:
                fn += 1
        
        # 计算该标签的精确率和召回率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # 计算该标签的 F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        label_f1s.append(f1)
    
    # 返回所有标签 F1 的平均值
    macro_f1 = sum(label_f1s) / len(label_f1s) if label_f1s else 0.0
    return macro_f1


def compute_example_f1(
    predictions: Sequence[Sequence[str]],
    gold_labels: Sequence[Sequence[str]],
) -> float:
    """
    计算 Example-F1 分数（也称为 Sample-F1）。
    
    Example-F1: 每个样本的 F1 分数的平均值。
    
    Args:
        predictions: 预测标签列表的列表（每个样本的预测标签）
        gold_labels: 真实标签列表的列表（每个样本的真实标签）
        
    Returns:
        Example-F1 分数（0-1之间）
    """
    example_f1s = []
    
    for pred, gold in zip(predictions, gold_labels):
        pred_set = set(pred)
        gold_set = set(gold)
        
        if not pred_set and not gold_set:
            # 如果预测和真实标签都为空，F1 为 1.0
            f1 = 1.0
        elif not pred_set or not gold_set:
            # 如果其中一个为空，F1 为 0.0
            f1 = 0.0
        else:
            tp = len(pred_set.intersection(gold_set))
            precision = tp / len(pred_set) if pred_set else 0.0
            recall = tp / len(gold_set) if gold_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        example_f1s.append(f1)
    
    # 返回所有样本 F1 的平均值
    example_f1 = sum(example_f1s) / len(example_f1s) if example_f1s else 0.0
    return example_f1


def compute_metrics(
    predictions: Sequence[Sequence[str]],
    gold_labels: Sequence[Sequence[str]],
    all_labels: Sequence[str],
) -> Dict[str, float]:
    """
    计算所有评估指标（micro-f1, macro-f1, example-f1）。
    
    Args:
        predictions: 预测标签列表的列表（每个样本的预测标签）
        gold_labels: 真实标签列表的列表（每个样本的真实标签）
        all_labels: 所有可能的标签列表
        
    Returns:
        包含所有指标的字典
    """
    micro_f1 = compute_micro_f1(predictions, gold_labels)
    macro_f1 = compute_macro_f1(predictions, gold_labels, all_labels)
    example_f1 = compute_example_f1(predictions, gold_labels)
    
    return {
        "micro-f1": round(micro_f1 * 100, 2),
        "macro-f1": round(macro_f1 * 100, 2),
        "example-f1": round(example_f1 * 100, 2),
    }

