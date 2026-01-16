"""
Naive memory bank implementation that mirrors the previous rag_cache behavior.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from memory.submodular import Exemplar


class NaiveMemoryBank:
    """
    Naive memory bank that mirrors the previous rag_cache behavior.
    """

    def __init__(self, B: int, k: int) -> None:
        self.B = B
        self.k = k
        self.S: List[Exemplar] = []

    def update(self, exemplar: Exemplar) -> bool:
        if len(self.S) < self.B:
            self.S.append(exemplar)
            return True
        min_idx = min(range(len(self.S)), key=lambda idx: self.S[idx].confidence)
        min_conf = self.S[min_idx].confidence
        if exemplar.confidence > min_conf:
            self.S[min_idx] = exemplar
            return True
        return False

    def retrieve(self, query_embedding: np.ndarray, k: Optional[int] = None) -> List[Exemplar]:
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


def _safe_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity, returning 0.0 for zero-norm vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))
