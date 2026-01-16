"""
Main entry for LLM-based multi-label classification evaluation.

This script runs LLM predictions on the test set and computes micro-f1,
macro-f1, and example-f1 metrics.
"""

import json
import logging
import os
from typing import Dict, List

import numpy as np

from config import parse_args, get_dataset_desc
from data_loader import load_tag_descriptions, load_test_file
from evaluation import compute_metrics
from llm_classifier import LLMClassifier
from prompt_builder import build_llm_prompt
from utils import get_text_embedding, get_text_embeddings, init_embedding_model
from l3r_confidence import aggregate_instance_confidence, format_l3r_validation
from memory.naive import NaiveMemoryBank
from memory.smb import SMBMemoryBank
from memory.submodular import Exemplar


def set_logging(output_dir: str) -> None:
    """
    Configure logging to console and file.

    Args:
        output_dir: Output directory where log.txt will be stored
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Log file path
    log_file = os.path.join(output_dir, "log.txt")

    # Log format
    log_format = "%(asctime)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Handlers: console + file (append mode)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
    ]

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,  # Avoid conflicts on repeated calls
    )


def main() -> None:
    """
    Run batch inference and evaluation.

    Steps:
    1. Load label descriptions and test data
    2. Initialize classifier
    3. Predict for each example (optionally limit sample count)
    4. Compute metrics (at sample 10, then every 100)
    5. Save predictions and metrics
    """
    args = parse_args()
    set_logging(output_dir=args.output_dir)

    # Initialize encoder model
    init_embedding_model(args.encoder_model)
    logging.info(f"Encoder model: {args.encoder_model}")

    logging.info("=" * 40)
    logging.info("Starting batch evaluation for multi-label classification")
    logging.info("=" * 40)
    logging.info(f"Dataset: {args.dataset}")

    # 1. Load label descriptions
    logging.info(f"Loading label descriptions: {args.tag_desc}")
    label_desc = load_tag_descriptions(args.tag_desc)
    label_space = list(label_desc.keys())
    if not label_space:
        raise RuntimeError(f"Label set is empty. Check {args.tag_desc}.")
    logging.info(f"Label count: {len(label_space)}")

    # Load dataset description if enabled
    dataset_desc = None
    if args.use_dataset_desc:
        dataset_desc = get_dataset_desc(args.dataset)
        logging.info(f"Using dataset description for: {args.dataset}")

    use_se_two_stage = args.dataset.lower() == "se"
    se_label_embeddings = None
    se_label_embeddings_matrix = None
    se_candidate_topk = min(args.se_candidate_topk, len(label_space))
    if use_se_two_stage:
        label_texts = []
        for label in label_space:
            desc = (label_desc.get(label) or "").strip()
            label_texts.append(desc if desc else label)
        logging.info(
            f"[se] Computing label embeddings: {len(label_texts)} items, "
            f"batch_size={args.se_label_embed_batch_size}"
        )
        se_label_embeddings = get_text_embeddings(
            label_texts, batch_size=args.se_label_embed_batch_size
        )
        if not se_label_embeddings or len(se_label_embeddings) != len(label_space):
            raise RuntimeError("se label embedding computation failed or size mismatch.")
        se_label_embeddings_matrix = np.array(se_label_embeddings, dtype=float)
        logging.info(f"[se] Stage-1 candidate labels: {se_candidate_topk}")

    def _select_se_candidate_labels(text: str) -> tuple[List[str], Dict[str, str]]:
        if se_label_embeddings_matrix is None or se_candidate_topk <= 0:
            return label_space, label_desc
        query_embedding = get_text_embedding(text)
        if not query_embedding:
            logging.warning("[se] Empty sample embedding. Using default candidate labels.")
            candidate_labels = label_space[:se_candidate_topk]
        else:
            query_vec = np.array(query_embedding, dtype=float)
            scores = se_label_embeddings_matrix.dot(query_vec)
            top_indices = np.argsort(scores)[::-1][:se_candidate_topk]
            candidate_labels = [label_space[int(idx)] for idx in top_indices]
        candidate_label_desc = {label: label_desc[label] for label in candidate_labels}
        return candidate_labels, candidate_label_desc
    
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
                f"[Consistency check] Low gold label coverage: {coverage:.2%}. "
                f"Dataset files may be mismatched, or labels differ by case/spacing."
            )

    # Limit test samples if specified
    total_samples = len(examples)
    if args.max_samples is not None and args.max_samples > 0:
        examples = examples[:args.max_samples]
        logging.info(f"Limiting samples: {len(examples)}/{total_samples}")
    else:
        logging.info(f"Total test samples: {total_samples}")

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
        conf_type=args.conf_type,
    )
    logging.info("Using LLM classifier")

    # RAG mode configuration
    rag_mode = (args.mode == "rag")
    rag_cache_size = args.cache_size
    rag_warmup = args.rag_warmup
    tau = getattr(args, "tau", 0.0)

    if args.bank_type == "smb":
        # Prefer --smb-k for backward compatibility, else use --rag-k
        rag_k = args.smb_k if args.smb_k is not None else args.rag_k
        _memory_bank = SMBMemoryBank(
            B=args.cache_size,
            k=rag_k,
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
        _memory_bank = NaiveMemoryBank(B=rag_cache_size, k=args.rag_k)

    if rag_mode:
        logging.info(f"[RAG mode] Enabled. Cache size: {rag_cache_size}, warmup: {rag_warmup}")
        logging.info(f"[MemoryBank] Type: {args.bank_type}, size: {len(_memory_bank.S)}")

    # 4. Predict for each example
    logging.info("Starting predictions...")
    predictions: List[List[str]] = []
    gold_labels: List[List[str]] = []
    did_l3r_validate = False
    
    for i, example in enumerate(examples, 1):
        candidate_label_space = label_space
        candidate_label_desc = label_desc
        if use_se_two_stage:
            candidate_label_space, candidate_label_desc = _select_se_candidate_labels(example.text)
        # RAG mode: use retrieval after warmup
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
                    f"[RAG] Sample #{example.idx}: retrieved {len(rag_examples)} examples "
                    f"(bank={args.bank_type}, size={len(_memory_bank.S)})"
                )

        # RAG mode after warmup: dual-channel prediction
        if rag_mode and use_rag_for_prediction:
            # Channel 1: prompt with RAG for final evaluation labels
            prompt_with_rag = build_llm_prompt(
                text=example.text,
                label_to_desc=candidate_label_desc,
                use_label_desc=args.use_label_desc,
                rag_examples=rag_examples,
                dataset_desc=dataset_desc,
            )
            prediction_result_rag = classifier.predict(
                prompt=prompt_with_rag,
                label_space=candidate_label_space,
                return_logprobs=False,  # Avoid logprobs to save cost
            )

            # Handle Channel 1 predictions for metrics
            if isinstance(prediction_result_rag, dict):
                predicted_labels = prediction_result_rag.get("labels", [])
            else:
                predicted_labels = prediction_result_rag
            
            predictions.append(predicted_labels)
            gold_labels.append(example.labels)
            logging.info(
                f"[RAG] [{i}/{len(examples)}] Sample #{example.idx}: "
                f"Channel 1 (with RAG) predicted {len(predicted_labels)} labels: "
                f"{predicted_labels[:5]}"
            )

            # Channel 2: prompt without RAG for confidence and update decisions
            prompt_without_rag = build_llm_prompt(
                text=example.text,
                label_to_desc=candidate_label_desc,
                use_label_desc=args.use_label_desc,
                rag_examples=None,  # No RAG examples
                dataset_desc=dataset_desc,
            )
            prediction_result_base = classifier.predict(
                prompt=prompt_without_rag,
                label_space=candidate_label_space,
                return_logprobs=True,  # Needed for confidence
            )

            # Handle Channel 2 predictions for confidence
            if isinstance(prediction_result_base, dict):
                base_predicted_labels = prediction_result_base.get("labels", [])
                confidences = prediction_result_base.get("confidences", {})
            else:
                base_predicted_labels = prediction_result_base
                confidences = {}

            # Compute sample confidence from non-RAG predictions
            if confidences and base_predicted_labels:
                label_confidences = [confidences.get(label, 0.0) for label in base_predicted_labels]
                sample_confidence = aggregate_instance_confidence(
                    label_confidences,
                    mode=args.l3r_agg,
                    top_m=args.l3r_top_m,
                )
                logging.debug(
                    f"[RAG] Sample #{example.idx}: Channel 2 (no RAG) predicted "
                    f"{len(base_predicted_labels)} labels: {base_predicted_labels[:5]}, "
                    f"confidence: {sample_confidence:.3f}"
                )
            else:
                sample_confidence = 0.0
                logging.warning(
                    f"Sample #{example.idx}: missing confidence, using default 0.0"
                )

            if args.conf_type == "l3r" and args.l3r_validate and not did_l3r_validate:
                logging.info(format_l3r_validation(confidences, sample_confidence))
                did_l3r_validate = True
            
            # Use confidence to decide whether to update MemoryBank.
            # MemoryBank uses Channel 2 (no RAG) predictions for consistency.
            # Final evaluation uses Channel 1 (with RAG) predictions.
            current_embedding = get_text_embedding(example.text)
            did_update = False
            if current_embedding and sample_confidence >= tau:
                exemplar = Exemplar(
                    text=example.text,
                    pred_labels=base_predicted_labels,  # Channel 2 predictions
                    confidence=sample_confidence,  # Channel 2 confidence
                    embedding=np.array(current_embedding, dtype=float),
                )
                before_len = len(_memory_bank.S)
                update_result = _memory_bank.update(exemplar)
                after_len = len(_memory_bank.S)
                did_update = bool(update_result) if update_result is not None else (after_len > before_len)

            logging.info(
                f"[RAG] MemoryBank update: Sample #{example.idx}, "
                f"updated={did_update}, conf={sample_confidence:.3f} (no-RAG prediction), "
                f"tau={tau:.3f}, size={len(_memory_bank.S)}"
            )

        else:
            # Base mode or before warmup: single prediction
            prompt = build_llm_prompt(
                text=example.text,
                label_to_desc=candidate_label_desc,
                use_label_desc=args.use_label_desc,
                rag_examples=None,
                dataset_desc=dataset_desc,
            )

            # Predict (RAG mode needs logprobs before warmup)
            return_logprobs = rag_mode
            prediction_result = classifier.predict(
                prompt=prompt,
                label_space=candidate_label_space,
                return_logprobs=return_logprobs,
            )

            # Handle prediction result
            if isinstance(prediction_result, dict):
                predicted_labels = prediction_result.get("labels", [])
                confidences = prediction_result.get("confidences", {})
            else:
                # Backward compatibility
                predicted_labels = prediction_result
                confidences = {}

            predictions.append(predicted_labels)
            gold_labels.append(example.labels)

            logging.info(
                f"[{i}/{len(examples)}] Sample #{example.idx}: predicted "
                f"{len(predicted_labels)} labels: {predicted_labels[:5]}"
            )

            # RAG mode: compute confidence and update cache before warmup
            if rag_mode:
                # Use aggregated confidence across predicted labels
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
                        f"Sample #{example.idx}: missing confidence, using default 0.0"
                    )

                if args.conf_type == "l3r" and args.l3r_validate and not did_l3r_validate:
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
                    f"[RAG] MemoryBank update: Sample #{example.idx}, "
                    f"updated={did_update}, conf={sample_confidence:.3f}, "
                    f"tau={tau:.3f}, size={len(_memory_bank.S)}"
                )

        # Log metrics at sample 10
        if i == 10:
            current_metrics = compute_metrics(
                predictions=predictions,
                gold_labels=gold_labels,
                all_labels=label_space,
            )
            logging.info(f"[Metrics] Processed {i}/{len(examples)} samples:")
            for key, value in current_metrics.items():
                logging.info(f"  {key}: {value:.2f}%")

        # Log metrics every 100 samples (skip 10 because already logged)
        elif i > 10 and i % 100 == 0:
            current_metrics = compute_metrics(
                predictions=predictions,
                gold_labels=gold_labels,
                all_labels=label_space,
            )
            logging.info(f"[Metrics] Processed {i}/{len(examples)} samples:")
            for key, value in current_metrics.items():
                logging.info(f"  {key}: {value:.2f}%")

    # 5. Compute final metrics
    logging.info("Computing final metrics...")
    final_metrics = compute_metrics(
        predictions=predictions,
        gold_labels=gold_labels,
        all_labels=label_space,
    )

    # 6. Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_output = os.path.join(args.output_dir, "metrics.json")
    logging.info(f"Saving metrics: {metrics_output}")
    with open(metrics_output, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    # 7. Save predictions
    pred_output = os.path.join(args.output_dir, "predictions.csv")
    logging.info(f"Saving predictions: {pred_output}")
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
    logging.info("Evaluation results:")
    logging.info(json.dumps(final_metrics, ensure_ascii=False, indent=2))
    logging.info("=" * 40)


if __name__ == "__main__":
    main()
