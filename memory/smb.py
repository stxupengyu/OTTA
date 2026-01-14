"""
Submodular Memory Bank (SMB) with online greedy updates.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from memory.submodular import Exemplar, compute_F


class SMBMemoryBank:
    """
    SMB memory bank with online greedy replacement.

    Maintains a set S with |S|<=B and a candidate_pool (sliding window of size W)
    that approximates U_t for F_div.
    """

    def __init__(
        self,
        B: int,
        k: int,
        lambdas: Mapping[str, float],
        epsilon: float,
        candidate_pool_strategy: str,
        W: int,
    ) -> None:
        """
        Args:
            B: Memory bank capacity.
            k: Default number of neighbors to retrieve.
            lambdas: Weights for F(S) terms (lambda1, lambda2, lambda3).
            epsilon: Epsilon used in F_cov.
            candidate_pool_strategy: Strategy name (currently only "sliding_window").
            W: Sliding window size for candidate_pool.
        """
        self.B = B
        self.k = k
        self.lambdas = dict(lambdas)
        self.epsilon = epsilon
        self.candidate_pool_strategy = candidate_pool_strategy
        self.W = W

        self.S: List[Exemplar] = []
        self.candidate_pool: List[Exemplar] = []
        self.label_counts: Dict[str, int] = {}

    def update(self, exemplar: Exemplar) -> None:
        """
        Online greedy update for S.

        If |S|<B, insert exemplar. Otherwise, find j in S maximizing
        F((S\{j})∪{i}) and replace only if improvement > 0.
        """
        self._update_candidate_pool(exemplar)
        self._ensure_label_counts(exemplar)

        if len(self.S) < self.B:
            self.S.append(exemplar)
            self._increment_label_counts(exemplar)
            return

        current_label_set = set(self.label_counts.keys())
        best_gain = 0.0
        best_idx: Optional[int] = None

        # Equation: choose j that maximizes F((S\{j})∪{i})
        for idx, old in enumerate(self.S):
            S_with = self.S[:idx] + [exemplar] + self.S[idx + 1 :]
            gain = compute_F(
                S_with,
                self.candidate_pool,
                current_label_set,
                self.lambdas,
                self.epsilon,
            ) - compute_F(
                self.S,
                self.candidate_pool,
                current_label_set,
                self.lambdas,
                self.epsilon,
            )
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx is not None and best_gain > 0.0:
            removed = self.S[best_idx]
            self._decrement_label_counts(removed)
            self.S[best_idx] = exemplar
            self._increment_label_counts(exemplar)

    def retrieve(self, query_embedding: np.ndarray, k: Optional[int] = None) -> List[Exemplar]:
        """
        Retrieve k nearest neighbors from S by cosine similarity.
        """
        if k is None:
            k = self.k
        if not self.S:
            return []

        scored = []
        for ex in self.S:
            sim = _safe_cosine_sim(query_embedding, ex.embedding)
            scored.append((sim, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:k]]

    def _update_candidate_pool(self, exemplar: Exemplar) -> None:
        """
        Maintain candidate_pool as a sliding window of the last W exemplars.

        This explicitly approximates U_t for F_div by a windowed candidate_pool.
        """
        if self.candidate_pool_strategy != "sliding_window":
            raise ValueError(f"Unsupported candidate_pool_strategy: {self.candidate_pool_strategy}")
        if self.W <= 0:
            return
        self.candidate_pool.append(exemplar)
        if len(self.candidate_pool) > self.W:
            self.candidate_pool.pop(0)

    def _ensure_label_counts(self, exemplar: Exemplar) -> None:
        for label in exemplar.pred_labels:
            if label not in self.label_counts:
                self.label_counts[label] = 0

    def _increment_label_counts(self, exemplar: Exemplar) -> None:
        for label in exemplar.pred_labels:
            self.label_counts[label] = self.label_counts.get(label, 0) + 1

    def _decrement_label_counts(self, exemplar: Exemplar) -> None:
        for label in exemplar.pred_labels:
            if label not in self.label_counts:
                continue
            self.label_counts[label] -= 1
            if self.label_counts[label] <= 0:
                del self.label_counts[label]


def _safe_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity, returning 0.0 for zero-norm vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))
