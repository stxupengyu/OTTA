"""
L3R (Label-set Local Likelihood Ratio) confidence computation.
"""

import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple


class LabelPrefixTrieNode:
    def __init__(self) -> None:
        self.children: Dict[str, "LabelPrefixTrieNode"] = {}
        self.is_terminal = False


class LabelPrefixTrie:
    def __init__(self) -> None:
        self.root = LabelPrefixTrieNode()

    def add(self, tokens: Sequence[str]) -> None:
        node = self.root
        for tok in tokens:
            if tok not in node.children:
                node.children[tok] = LabelPrefixTrieNode()
            node = node.children[tok]
        node.is_terminal = True

    def get_competitor_tokens(self, prefix_tokens: Sequence[str], self_token: str) -> List[str]:
        node = self.root
        for tok in prefix_tokens:
            node = node.children.get(tok)
            if node is None:
                return []
        return [tok for tok in node.children.keys() if tok != self_token]


def build_label_prefix_trie(
    labels: Sequence[str],
    tokenizer: Callable[[str], Sequence[str]],
) -> LabelPrefixTrie:
    trie = LabelPrefixTrie()
    for label in labels:
        tokens = list(tokenizer(label))
        if tokens:
            trie.add(tokens)
    return trie


def aggregate_instance_confidence(
    label_scores: Sequence[float],
    mode: str = "mean",
    top_m: Optional[int] = None,
) -> float:
    if not label_scores:
        return 0.0
    if mode == "max":
        return max(label_scores)
    if mode == "top-m":
        if top_m is None or top_m <= 0:
            top_m = len(label_scores)
        top_m = min(len(label_scores), top_m)
        return sum(sorted(label_scores, reverse=True)[:top_m]) / top_m
    return sum(label_scores) / len(label_scores)


def compute_l3r_per_label(
    predicted_labels: Sequence[str],
    label_space: Sequence[str],
    token_logprobs: Sequence[Dict],
    content: str,
    trie: LabelPrefixTrie,
    eps: float,
    alpha: float,
    tokenizer: Callable[[str], Sequence[str]],
) -> Dict[str, float]:
    label_scores: Dict[str, float] = {}
    if not token_logprobs or not predicted_labels:
        return label_scores

    token_positions = _build_token_positions(token_logprobs)

    for label in predicted_labels:
        steps = _find_label_token_steps(label, content, token_positions)
        if not steps:
            label_scores[label] = 0.0
            continue

        label_tokens = [token for _, token in steps]
        llrs: List[float] = []
        entropies: List[float] = []

        for idx_in_label, (step_idx, token) in enumerate(steps):
            prefix_tokens = label_tokens[:idx_in_label]
            competitor_tokens = trie.get_competitor_tokens(prefix_tokens, token)

            step = token_logprobs[step_idx]
            p_self = _get_token_prob(step, token)
            p_comp = 0.0
            for comp_tok in competitor_tokens:
                p_comp += _get_token_prob(step, comp_tok)

            llr = math.log((p_self + eps) / (p_comp + eps))
            llrs.append(llr)

            denom = p_self + p_comp + eps
            q_self = p_self / denom
            q_comp = 1.0 - q_self
            if q_self <= 0.0 or q_self >= 1.0:
                entropy = 0.0
            else:
                entropy = -((q_self * math.log(q_self)) + (q_comp * math.log(q_comp)))
            entropies.append(entropy)

        weights = _softmax(entropies, alpha)
        score = sum(w * llr for w, llr in zip(weights, llrs))
        label_scores[label] = score

    return label_scores


def _build_token_vocab(token_logprobs: Sequence[Dict]) -> List[str]:
    vocab = set()
    for step in token_logprobs:
        token = step.get("token")
        if token:
            vocab.add(token)
        top_logprobs = step.get("top_logprobs") or {}
        for tok in top_logprobs.keys():
            if tok:
                vocab.add(tok)
    return sorted(vocab, key=len, reverse=True)


def _tokenize_with_vocab(label: str, vocab_sorted: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    if not label:
        return tokens

    i = 0
    while i < len(label):
        match = None
        for tok in vocab_sorted:
            if label.startswith(tok, i):
                match = tok
                break
        if match is None:
            tokens.append(label[i])
            i += 1
        else:
            tokens.append(match)
            i += len(match)
    return tokens


def build_label_tokenizer(token_logprobs: Sequence[Dict]) -> Callable[[str], Sequence[str]]:
    vocab_sorted = _build_token_vocab(token_logprobs)

    def _tokenizer(label: str) -> Sequence[str]:
        return _tokenize_with_vocab(label, vocab_sorted)

    return _tokenizer


def _build_token_positions(token_logprobs: Sequence[Dict]) -> List[Tuple[int, int, int, str]]:
    positions: List[Tuple[int, int, int, str]] = []
    accumulated_pos = 0
    for idx, step in enumerate(token_logprobs):
        token = step.get("token")
        if token is None:
            continue
        start = accumulated_pos
        end = accumulated_pos + len(token)
        positions.append((idx, start, end, token))
        accumulated_pos = end
    return positions


def _find_label_token_steps(
    label: str,
    content: str,
    token_positions: Sequence[Tuple[int, int, int, str]],
) -> List[Tuple[int, str]]:
    if not label or not content:
        return []

    match = re.search(r"\"{}\"".format(re.escape(label)), content)
    if not match:
        return []

    label_start = match.start() + 1
    label_end = match.end() - 1

    steps: List[Tuple[int, str]] = []
    for idx, start, end, token in token_positions:
        if not (end <= label_start or start >= label_end):
            steps.append((idx, token))
    return steps


def _get_token_prob(step: Dict, token: str) -> float:
    top_logprobs = step.get("top_logprobs") or {}
    if token in top_logprobs:
        return math.exp(top_logprobs[token])
    if token == step.get("token") and step.get("logprob") is not None:
        return math.exp(step["logprob"])
    return 0.0


def _softmax(values: Sequence[float], alpha: float) -> List[float]:
    if not values:
        return []
    scaled = [alpha * v for v in values]
    max_val = max(scaled)
    exps = [math.exp(v - max_val) for v in scaled]
    denom = sum(exps)
    if denom == 0.0:
        return [1.0 / len(values)] * len(values)
    return [v / denom for v in exps]


def format_l3r_validation(
    label_scores: Dict[str, float],
    instance_confidence: float,
) -> str:
    parts = ["L3R validation:"]
    for label, score in label_scores.items():
        parts.append(f"  {label}: {score:.6f}")
    parts.append(f"  instance_confidence: {instance_confidence:.6f}")
    return "\n".join(parts)
