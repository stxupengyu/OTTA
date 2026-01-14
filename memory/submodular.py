"""
Submodular objective for memory bank selection (SMB).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np


@dataclass
class Exemplar:
    """
    Stored exemplar used in the memory bank.

    Attributes:
        text: Raw input text.
        pred_labels: Predicted label strings.
        confidence: Instance-level confidence (L3R aggregation).
        embedding: Vector representation of the text.
    """

    text: str
    pred_labels: List[str]
    confidence: float
    embedding: np.ndarray


def _safe_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity, returning 0.0 for zero-norm vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def compute_f_cov(S: Sequence[Exemplar], label_set: Set[str], epsilon: float) -> float:
    """
    Compute F_cov(S) per the coverage definition.

    F_cov(S) = sum_{l in label_set} (1/sqrt(f_l(S)+epsilon)) * I[l is covered by S]
    """
    # f_l(S) = count of exemplars in S whose pred_labels contain l
    label_counts = {label: 0 for label in label_set}
    for ex in S:
        for lab in ex.pred_labels:
            if lab in label_counts:
                label_counts[lab] += 1

    total = 0.0
    for label in label_set:
        count = label_counts[label]
        if count > 0:
            total += 1.0 / math.sqrt(count + epsilon)
    return total


def compute_f_div(S: Sequence[Exemplar], candidate_pool: Sequence[Exemplar]) -> float:
    """
    Compute F_div(S) with facility location over candidate_pool as U_t approximation.

    F_div(S) = sum_{u in candidate_pool} max_{i in S} cosine_sim(u.embedding, i.embedding)

    Note: candidate_pool approximates U_t (e.g., a sliding window of recent samples).
    """
    if not candidate_pool:
        return 0.0

    total = 0.0
    for u in candidate_pool:
        best = 0.0
        for i in S:
            sim = _safe_cosine_sim(u.embedding, i.embedding)
            if sim > best:
                best = sim
        total += best
    return total


def compute_f_qual(S: Sequence[Exemplar]) -> float:
    """
    Compute F_qual(S) as sum of exemplar confidences.

    F_qual(S) = sum_{i in S} i.confidence
    """
    return sum(ex.confidence for ex in S)


def compute_F(
    S: Sequence[Exemplar],
    candidate_pool: Sequence[Exemplar],
    label_set: Set[str],
    lambdas: Mapping[str, float],
    epsilon: float,
) -> float:
    """
    Compute F(S) = lambda1*F_cov(S) + lambda2*F_div(S) + lambda3*F_qual(S).
    """
    lambda1 = lambdas.get("lambda1", 0.0)
    lambda2 = lambdas.get("lambda2", 0.0)
    lambda3 = lambdas.get("lambda3", 0.0)

    f_cov = compute_f_cov(S, label_set, epsilon)
    f_div = compute_f_div(S, candidate_pool)
    f_qual = compute_f_qual(S)

    return (lambda1 * f_cov) + (lambda2 * f_div) + (lambda3 * f_qual)


def marginal_gain(
    i: Exemplar,
    S: Sequence[Exemplar],
    candidate_pool: Sequence[Exemplar],
    label_set: Set[str],
    lambdas: Mapping[str, float],
    epsilon: float,
) -> float:
    """
    Compute the marginal gain: F(S ∪ {i}) - F(S).
    """
    S_with = list(S) + [i]
    return compute_F(S_with, candidate_pool, label_set, lambdas, epsilon) - compute_F(
        S, candidate_pool, label_set, lambdas, epsilon
    )
