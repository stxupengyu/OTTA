"""
基于 LLM 的多标签分类评估主程序。

该脚本使用大语言模型（LLM）对测试集进行多标签分类预测，
并计算 micro-f1, macro-f1, example-f1 等评估指标。
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
    设置日志记录，同时输出到控制台和文件。
    
    Args:
        output_dir: 输出目录，日志文件将保存在此目录下的 log.txt
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 日志文件路径
    log_file = os.path.join(output_dir, "log.txt")
    
    # 配置日志格式
    log_format = "%(asctime)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 创建处理器：控制台和文件（追加模式）
    handlers = [
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler(log_file, mode='a', encoding='utf-8')  # 文件输出（追加模式）
    ]
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True  # 强制重新配置，避免多次调用时的冲突
    )


def main() -> None:
    """
    主函数：执行批量推理和评估流程。
    
    流程：
    1. 加载标签描述和测试数据
    2. 初始化分类器
    3. 对每个样本进行预测（支持限制样本数量）
    4. 计算评估指标（在第10个样本时，之后每100个样本输出一次）
    5. 保存预测结果和指标
    """
    args = parse_args()
    set_logging(output_dir=args.output_dir)
    
    logging.info("=" * 40)
    logging.info("开始批量评估多标签分类任务")
    logging.info("=" * 40)
    logging.info(f"数据集: {args.dataset}")
    
    # 1. 加载标签描述
    logging.info(f"加载标签描述: {args.tag_desc}")
    label_desc = load_tag_descriptions(args.tag_desc)
    label_space = list(label_desc.keys())
    if not label_space:
        raise RuntimeError(f"标签集合为空，请检查 {args.tag_desc}")
    logging.info(f"标签数量: {len(label_space)}")
    
    # 2. 加载测试数据
    logging.info(f"加载测试数据: {args.test_file}")
    examples = load_test_file(args.test_file)
    if not examples:
        raise RuntimeError(f"未能从 {args.test_file} 解析到样本。")
    
    # 基本一致性检查：gold labels 是否在 label_space 中
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
                f"[一致性检查] gold labels 覆盖率过低: {coverage:.2%}。"
                f"可能是数据集文件不匹配，或标签存在大小写/空格差异。"
            )
    
    # 限制测试样本数量（如果指定）
    total_samples = len(examples)
    if args.max_samples is not None and args.max_samples > 0:
        examples = examples[:args.max_samples]
        logging.info(f"限制测试样本数量: {len(examples)}/{total_samples}")
    else:
        logging.info(f"测试样本总数: {total_samples}")
    
    # 3. 初始化分类器
    logging.info(f"模式: {args.mode}")
    logging.info(f"模型类型: {args.model_type}")
    logging.info(f"模型名称: {args.model_name}")
    logging.info(f"Base URL: {args.base_url or 'OpenAI默认'}")
    
    classifier = LLMClassifier(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        system_prompt=args.system_prompt,
        request_interval=args.request_interval,
        l3r_eps=args.l3r_eps,
        l3r_alpha=args.l3r_alpha,
    )
    logging.info("使用 LLM 分类器")
    
    # RAG 模式相关配置
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
        logging.info(f"[RAG模式] 已启用，cache大小: {rag_cache_size}，warmup: {rag_warmup}")
        logging.info(f"[MemoryBank] 类型: {args.bank_type}, size: {len(_memory_bank.S)}")
    
    # 4. 对每个样本进行预测
    logging.info("开始预测...")
    predictions: List[List[str]] = []
    gold_labels: List[List[str]] = []
    did_l3r_validate = False
    
    for i, example in enumerate(examples, 1):
        # RAG模式：在warmup之后，使用RAG检索相似样本
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
                    f"[RAG] 样本 #{example.idx}: 检索到 {len(rag_examples)} 个相似样本用于RAG "
                    f"(bank={args.bank_type}, size={len(_memory_bank.S)})"
                )
        
        # RAG模式且warmup之后：进行两次预测
        if rag_mode and use_rag_for_prediction:
            # 使用包含示例的 prompt（RAG），用于评价指标与置信度计算（同一次调用）
            prompt_with_rag = build_llm_prompt(
                text=example.text,
                label_to_desc=label_desc,
                use_label_desc=args.use_label_desc,
                rag_examples=rag_examples,
            )
            prediction_result_rag = classifier.predict(
                prompt=prompt_with_rag,
                label_space=label_space,
                return_logprobs=True,  # 需要logprob来计算置信度
            )
            
            # 处理预测结果（用于评价指标与置信度）
            if isinstance(prediction_result_rag, dict):
                predicted_labels = prediction_result_rag.get("labels", [])
                confidences = prediction_result_rag.get("confidences", {})
            else:
                predicted_labels = prediction_result_rag
                confidences = {}
            
            predictions.append(predicted_labels)
            gold_labels.append(example.labels)
            logging.info(f"[RAG] [{i}/{len(examples)}] 样本 #{example.idx}: 预测 {len(predicted_labels)} 个标签: {predicted_labels[:5]}")
            
            # 计算样本置信度：取预测的k个标签的置信度均值
            if confidences and predicted_labels:
                label_confidences = [confidences.get(label, 0.0) for label in predicted_labels]
                sample_confidence = aggregate_instance_confidence(
                    label_confidences,
                    mode=args.l3r_agg,
                    top_m=args.l3r_top_m,
                )
            else:
                sample_confidence = 0.0
                logging.warning(f"样本 #{example.idx}: 无法获取置信度，使用默认值 0.0")

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
                f"[RAG] MemoryBank 更新: 样本 #{example.idx}, "
                f"updated={did_update}, conf={sample_confidence:.3f}, "
                f"tau={tau:.3f}, size={len(_memory_bank.S)}"
            )
        
        else:
            # 基础模式或warmup之前：只进行一次预测
            # 构建提示词
            prompt = build_llm_prompt(
                text=example.text,
                label_to_desc=label_desc,
                use_label_desc=args.use_label_desc,
                rag_examples=None,
            )
            
            # 进行预测
            # RAG模式下需要获取logprob和置信度（warmup之前）
            return_logprobs = rag_mode
            prediction_result = classifier.predict(
                prompt=prompt,
                label_space=label_space,
                return_logprobs=return_logprobs,
            )
            
            # 处理预测结果
            if isinstance(prediction_result, dict):
                predicted_labels = prediction_result.get("labels", [])
                confidences = prediction_result.get("confidences", {})
            else:
                # 向后兼容：如果不是字典，直接使用列表
                predicted_labels = prediction_result
                confidences = {}
            
            predictions.append(predicted_labels)
            gold_labels.append(example.labels)
            
            logging.info(f"[{i}/{len(examples)}] 样本 #{example.idx}: 预测 {len(predicted_labels)} 个标签: {predicted_labels[:5]}")
            
            # RAG模式：计算样本置信度并更新缓存（warmup之前）
            if rag_mode:
                # 计算样本置信度：取预测的k个标签的置信度均值
                if confidences and predicted_labels:
                    label_confidences = [confidences.get(label, 0.0) for label in predicted_labels]
                    sample_confidence = aggregate_instance_confidence(
                        label_confidences,
                        mode=args.l3r_agg,
                        top_m=args.l3r_top_m,
                    )
                else:
                    sample_confidence = 0.0
                    logging.warning(f"样本 #{example.idx}: 无法获取置信度，使用默认值 0.0")

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
                    f"[RAG] MemoryBank 更新: 样本 #{example.idx}, "
                    f"updated={did_update}, conf={sample_confidence:.3f}, "
                    f"tau={tau:.3f}, size={len(_memory_bank.S)}"
                )
        
        # 在第10个样本时输出指标
        if i == 10:
            current_metrics = compute_metrics(
                predictions=predictions,
                gold_labels=gold_labels,
                all_labels=label_space,
            )
            logging.info(f"[实时指标] 已处理 {i}/{len(examples)} 个样本:")
            for key, value in current_metrics.items():
                logging.info(f"  {key}: {value:.2f}%")
        
        # 每100个样本输出一次指标（跳过第10个，因为已经输出过了）
        elif i > 10 and i % 100 == 0:
            current_metrics = compute_metrics(
                predictions=predictions,
                gold_labels=gold_labels,
                all_labels=label_space,
            )
            logging.info(f"[实时指标] 已处理 {i}/{len(examples)} 个样本:")
            for key, value in current_metrics.items():
                logging.info(f"  {key}: {value:.2f}%")
    
    # 5. 计算最终评估指标
    logging.info("计算最终评估指标...")
    final_metrics = compute_metrics(
        predictions=predictions,
        gold_labels=gold_labels,
        all_labels=label_space,
    )
    
    # 6. 保存评估指标
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_output = os.path.join(args.output_dir, "metrics.json")
    logging.info(f"保存评估指标: {metrics_output}")
    with open(metrics_output, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    
    # 7. 保存预测结果
    pred_output = os.path.join(args.output_dir, "predictions.csv")
    logging.info(f"保存预测结果: {pred_output}")
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
    logging.info("评估结果:")
    logging.info(json.dumps(final_metrics, ensure_ascii=False, indent=2))
    logging.info("=" * 40)


if __name__ == "__main__":
    main()
