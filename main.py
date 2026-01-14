"""
Main entry for LLM-based multi-label evaluation.

This script runs multi-label classification on a test set and computes
micro-f1, macro-f1, and example-f1.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np

from config import parse_args
from data_loader import load_tag_descriptions, load_test_file
from evaluation import compute_metrics
from llm_classifier import LLMClassifier
from prompt_builder import build_llm_prompt
from utils import get_text_embedding
from l3r_confidence import aggregate_instance_confidence, format_l3r_validation
from memory.smb import SMBMemoryBank
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
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def set_logging(output_dir: str) -> None:
    """
    Configure logging to both console and file.

    Args:
        output_dir: Output directory; log file is saved as log.txt
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Log file path
    log_file = os.path.join(output_dir, "log.txt")
    
    # Log formatting
    log_format = "%(asctime)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Handlers: console and file (append mode)
    handlers = [
        logging.StreamHandler(),  # Console
        logging.FileHandler(log_file, mode='a', encoding='utf-8')  # File (append)
    ]
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True  # Reconfigure to avoid duplicate handlers
    )


def main() -> None:
    """
    Main flow for batch inference and evaluation.

    Steps:
    1. Load label descriptions and test data
    2. Initialize classifier
    3. Predict for each sample (optionally limit samples)
    4. Compute metrics (after 10 samples, then every 100)
    5. Save predictions and metrics
    """
    args = parse_args()
    set_logging(output_dir=args.output_dir)
    
    logging.info("=" * 40)
    logging.info("Starting batch evaluation for multi-label classification")
    logging.info("=" * 40)
    logging.info(f"Dataset: {args.dataset}")
    
    # 1. Load label descriptions
    logging.info(f"Loading label descriptions: {args.tag_desc}")
    label_desc = load_tag_descriptions(args.tag_desc)
    label_space = list(label_desc.keys())
    if not label_space:
        raise RuntimeError(f"Empty label set. Check: {args.tag_desc}")
    logging.info(f"Label count: {len(label_space)}")
    
    # 2. Load test data
    logging.info(f"Loading test data: {args.test_file}")
    examples = load_test_file(args.test_file)
    if not examples:
        raise RuntimeError(f"No samples parsed from {args.test_file}.")
    
    # Basic consistency check: gold labels in label_space
    label_set = set(label_space)
    gold_total = 0
    gold_in_space = 0
    for ex in examples:
        for lab in ex.labels:
            gold_total += 1
            if lab in label_set:
                gold_in_space += 1
    if gold_total > 0:
        coverage = gold_in_space / gold_total
        if coverage < 0.8:
            logging.warning(
                "[Consistency check] Low gold label coverage: %.2f%%. "
                "Possible dataset mismatch or label casing/spacing issues.",
                coverage * 100,
            )
    
    # Limit test samples if requested
    total_samples = len(examples)
    if args.max_samples is not None and args.max_samples > 0:
        examples = examples[:args.max_samples]
        logging.info("Limiting test samples: %s/%s", len(examples), total_samples)
    else:
        logging.info("Total test samples: %s", total_samples)
    
    # 3. Initialize classifier
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Model type: {args.model_type}")
    logging.info(f"Model name: {args.model_name}")
    logging.info(f"Base URL: {args.base_url or 'OpenAI default'}")
    
    classifier = LLMClassifier(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        system_prompt=args.system_prompt,
        request_interval=args.request_interval,
        l3r_eps=args.l3r_eps,
        l3r_alpha=args.l3r_alpha,
    )
    logging.info("Using LLM classifier")
    
    # RAG configuration
    rag_mode = (args.mode == "rag")
    rag_cache_size = 100
    rag_warmup = 200
    tau = getattr(args, "tau", 0.0)

    if args.bank_type == "smb":
        _memory_bank = SMBMemoryBank(
            B=args.smb_B,
            k=args.smb_k,
            lambdas={
                "lambda1": args.smb_lambda1,
                "lambda2": args.smb_lambda2,
                "lambda3": args.smb_lambda3,
            },
            epsilon=args.smb_epsilon,
            candidate_pool_strategy="sliding_window",
            W=args.smb_W,
        )
    else:
        _memory_bank = NaiveMemoryBank(B=rag_cache_size, k=10)
    
    if rag_mode:
        logging.info(
            "[RAG] enabled, cache size: %s, warmup: %s",
            rag_cache_size,
            rag_warmup,
        )
        logging.info("[MemoryBank] type: %s, size: %s", args.bank_type, len(_memory_bank.S))
    
    # 4. Predict each sample
    logging.info("Running predictions...")
    predictions: List[List[str]] = []
    gold_labels: List[List[str]] = []
    did_l3r_validate = False
    
    for i, example in enumerate(examples, 1):
        # RAG mode: after warmup, retrieve similar samples
        rag_examples = None
        use_rag_for_prediction = False
        if rag_mode and i > rag_warmup and len(_memory_bank.S) > 0:
            current_embedding = get_text_embedding(example.text)
            if current_embedding:
                query_embedding = np.array(current_embedding, dtype=float)
                demos = _memory_bank.retrieve(query_embedding, k=_memory_bank.k)
                rag_examples = [
                    {
                        "text": ex.text,
                        "prediction": {"labels": ex.pred_labels},
                    }
                    for ex in demos
                ]
                use_rag_for_prediction = True
                logging.info(
                    "[RAG] sample #%s: retrieved %s similar examples "
                    "(bank=%s, size=%s)",
                    example.idx,
                    len(rag_examples),
                    args.bank_type,
                    len(_memory_bank.S),
                )
        
        # RAG mode after warmup: run two predictions
        if rag_mode and use_rag_for_prediction:
            # Prompt with RAG examples (for metrics and confidence)
            prompt_with_rag = build_llm_prompt(
                text=example.text,
                label_to_desc=label_desc,
                use_label_desc=args.use_label_desc,
                rag_examples=rag_examples,
            )
            prediction_result_rag = classifier.predict(
                prompt=prompt_with_rag,
                label_space=label_space,
                return_logprobs=True,  # logprobs needed for confidence
            )
            
            # Handle prediction result (metrics + confidence)
            if isinstance(prediction_result_rag, dict):
                predicted_labels = prediction_result_rag.get("labels", [])
                confidences = prediction_result_rag.get("confidences", {})
            else:
                predicted_labels = prediction_result_rag
                confidences = {}
            
            predictions.append(predicted_labels)
            gold_labels.append(example.labels)
            logging.info(
                "[RAG] [%s/%s] sample #%s: predicted %s labels: %s",
                i,
                len(examples),
                example.idx,
                len(predicted_labels),
                predicted_labels[:5],
            )
            
            # Compute sample confidence as mean of predicted label confidences
            if confidences and predicted_labels:
                label_confidences = [confidences.get(label, 0.0) for label in predicted_labels]
                sample_confidence = aggregate_instance_confidence(
                    label_confidences,
                    mode=args.l3r_agg,
                    top_m=args.l3r_top_m,
                )
            else:
                sample_confidence = 0.0
                logging.warning(
                    "sample #%s: confidence unavailable, using 0.0",
                    example.idx,
                )

            if args.l3r_validate and not did_l3r_validate:
                logging.info(format_l3r_validation(confidences, sample_confidence))
                did_l3r_validate = True
            
            current_embedding = get_text_embedding(example.text)
            did_update = False
            if current_embedding and sample_confidence >= tau:
                exemplar = Exemplar(
                    text=example.text,
                    pred_labels=predicted_labels,
                    confidence=sample_confidence,
                    embedding=np.array(current_embedding, dtype=float),
                )
                before_len = len(_memory_bank.S)
                update_result = _memory_bank.update(exemplar)
                after_len = len(_memory_bank.S)
                did_update = bool(update_result) if update_result is not None else (after_len > before_len)

            logging.info(
                "[RAG] MemoryBank update: sample #%s, updated=%s, conf=%.3f, "
                "tau=%.3f, size=%s",
                example.idx,
                did_update,
                sample_confidence,
                tau,
                len(_memory_bank.S),
            )
        
        else:
            # Base mode or before warmup: single prediction
            # Build prompt
            prompt = build_llm_prompt(
                text=example.text,
                label_to_desc=label_desc,
                use_label_desc=args.use_label_desc,
                rag_examples=None,
            )
            
            # Predict
            # In RAG mode we still need logprobs before warmup
            return_logprobs = rag_mode
            prediction_result = classifier.predict(
                prompt=prompt,
                label_space=label_space,
                return_logprobs=return_logprobs,
            )
            
            # Handle prediction results
            if isinstance(prediction_result, dict):
                predicted_labels = prediction_result.get("labels", [])
                confidences = prediction_result.get("confidences", {})
            else:
                # Backward compatibility: list output
                predicted_labels = prediction_result
                confidences = {}
            
            predictions.append(predicted_labels)
            gold_labels.append(example.labels)
            
            logging.info(
                "[%s/%s] sample #%s: predicted %s labels: %s",
                i,
                len(examples),
                example.idx,
                len(predicted_labels),
                predicted_labels[:5],
            )
            
            # RAG mode: compute confidence and update cache (before warmup)
            if rag_mode:
                # Compute confidence as mean of predicted label confidences
                if confidences and predicted_labels:
                    label_confidences = [confidences.get(label, 0.0) for label in predicted_labels]
                    sample_confidence = aggregate_instance_confidence(
                        label_confidences,
                        mode=args.l3r_agg,
                        top_m=args.l3r_top_m,
                    )
                else:
                    sample_confidence = 0.0
                    logging.warning(
                        "sample #%s: confidence unavailable, using 0.0",
                        example.idx,
                    )

                if args.l3r_validate and not did_l3r_validate:
                    logging.info(format_l3r_validation(confidences, sample_confidence))
                    did_l3r_validate = True
                
                current_embedding = get_text_embedding(example.text)
                did_update = False
                if current_embedding and sample_confidence >= tau:
                    exemplar = Exemplar(
                        text=example.text,
                        pred_labels=predicted_labels,
                        confidence=sample_confidence,
                        embedding=np.array(current_embedding, dtype=float),
                    )
                    before_len = len(_memory_bank.S)
                    update_result = _memory_bank.update(exemplar)
                    after_len = len(_memory_bank.S)
                    did_update = bool(update_result) if update_result is not None else (after_len > before_len)

                logging.info(
                    "[RAG] MemoryBank update: sample #%s, updated=%s, conf=%.3f, "
                    "tau=%.3f, size=%s",
                    example.idx,
                    did_update,
                    sample_confidence,
                    tau,
                    len(_memory_bank.S),
                )
        
        # Report metrics after 10 samples
        if i == 10:
            current_metrics = compute_metrics(
                predictions=predictions,
                gold_labels=gold_labels,
                all_labels=label_space,
            )
            logging.info("[Metrics] processed %s/%s samples:", i, len(examples))
            for key, value in current_metrics.items():
                logging.info("  %s: %.2f%%", key, value)
        
        # Report metrics every 100 samples after the first 10
        elif i > 10 and i % 100 == 0:
            current_metrics = compute_metrics(
                predictions=predictions,
                gold_labels=gold_labels,
                all_labels=label_space,
            )
            logging.info("[Metrics] processed %s/%s samples:", i, len(examples))
            for key, value in current_metrics.items():
                logging.info("  %s: %.2f%%", key, value)
    
    # 5. Final metrics
    logging.info("Computing final metrics...")
    final_metrics = compute_metrics(
        predictions=predictions,
        gold_labels=gold_labels,
        all_labels=label_space,
    )
    
    # 6. Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_output = os.path.join(args.output_dir, "metrics.json")
    logging.info("Saving metrics: %s", metrics_output)
    with open(metrics_output, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    
    # 7. Save predictions
    pred_output = os.path.join(args.output_dir, "predictions.csv")
    logging.info("Saving predictions: %s", pred_output)
    import csv
    with open(pred_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "gold_labels", "predicted_labels"])
        writer.writeheader()
        for example, pred in zip(examples, predictions):
            writer.writerow({
                "id": example.idx,
                "text": example.text,
                "gold_labels": ";".join(example.labels),
                "predicted_labels": ";".join(pred),
            })
    
    logging.info("=" * 40)
    logging.info("Final metrics:")
    logging.info(json.dumps(final_metrics, ensure_ascii=False, indent=2))
    logging.info("=" * 40)


if __name__ == "__main__":
    main()
