"""
配置和参数解析。
"""

import argparse
import os

from models import (
    DEFAULT_MODEL,
    DEFAULT_BASE_URL,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_SILICONFLOW_API_KEY,
)

# 项目根目录（绝对路径）
PROJECT_ROOT = "/Users/xupengyu/code/llm4mll"

# 数据集名称映射（参数值 -> 实际目录名）
DATASET_NAME_MAP = {
    "movie": "MOVIE",
    "aapd": "AAPD",
    "rcv": "RCV",
    "se": "StackExchange",
}


# LLM 模型类型到模型名称的映射
LLM_MODEL_NAME_MAP = {
    "gpt3.5": "gpt-3.5-turbo",
    "gpt4o": "gpt-4o",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
}

# LLM 模型类型到 API Base URL 的映射
LLM_BASE_URL_MAP = {
    "gpt3.5": None,  # None 表示使用 OpenAI 默认地址
    "gpt4o": None,
    "qwen2.5": "https://api.siliconflow.cn/v1",
}

# LLM 模型类型到 API Key 的映射
LLM_API_KEY_MAP = {
    "gpt3.5": DEFAULT_OPENAI_API_KEY,
    "gpt4o": DEFAULT_OPENAI_API_KEY,
    "qwen2.5": DEFAULT_SILICONFLOW_API_KEY,
}


def get_llm_model_name(model_type: str) -> str:
    """
    根据 LLM 模型类型获取模型名称。
    
    Args:
        model_type: LLM 模型类型（gpt3.5, gpt4o, qwen2.5）
        
    Returns:
        对应的模型名称
        
    Raises:
        ValueError: 如果模型类型不支持
    """
    model_type = model_type.lower()
    if model_type not in LLM_MODEL_NAME_MAP:
        raise ValueError(
            f"不支持的 LLM 模型类型: {model_type}。"
            f"支持的模型类型: {list(LLM_MODEL_NAME_MAP.keys())}"
        )
    return LLM_MODEL_NAME_MAP[model_type]


def get_llm_base_url(model_type: str) -> str:
    """
    根据 LLM 模型类型获取 API Base URL。
    
    Args:
        model_type: LLM 模型类型
        
    Returns:
        API Base URL（None 表示使用 OpenAI 默认地址）
    """
    model_type = model_type.lower()
    return LLM_BASE_URL_MAP.get(model_type, None)


def get_llm_api_key(model_type: str) -> str:
    """
    根据 LLM 模型类型获取默认 API Key。
    
    Args:
        model_type: LLM 模型类型
        
    Returns:
        API Key
    """
    model_type = model_type.lower()
    return LLM_API_KEY_MAP.get(model_type, DEFAULT_OPENAI_API_KEY)


def get_dataset_paths(dataset: str) -> tuple[str, str]:
    """
    根据数据集名称获取测试文件和标签描述文件的绝对路径。
    
    Args:
        dataset: 数据集名称（movie, aapd, rcv, se）
        
    Returns:
        (test_file_path, tag_desc_path) 元组，均为绝对路径
        
    Raises:
        ValueError: 如果数据集名称不支持
    """
    dataset = dataset.lower()
    if dataset not in DATASET_NAME_MAP:
        raise ValueError(
            f"不支持的数据集: {dataset}。"
            f"支持的数据集: {list(DATASET_NAME_MAP.keys())}"
        )
    
    dataset_dir = DATASET_NAME_MAP[dataset]
    data_dir = os.path.join(PROJECT_ROOT, "data", dataset_dir)
    test_file = os.path.join(data_dir, "test.txt")
    tag_desc = os.path.join(data_dir, "tag_description.csv")
    
    return test_file, tag_desc


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    
    Returns:
        包含所有命令行参数的命名空间对象
    """
    parser = argparse.ArgumentParser(description="基于 LLM 的多标签分类评估。")
    
    # 数据集参数
    parser.add_argument(
        "--dataset",
        type=str,
        default="movie",
        choices=["movie", "aapd", "rcv", "se"],
        help="数据集名称：movie, aapd, rcv, se（默认: movie）",
    )
    
    # 模型配置
    parser.add_argument(
        "--model-type",
        type=str,
        default="gpt3.5",
        choices=["gpt3.5", "gpt4o", "qwen2.5"],
        help="模型类型：gpt3.5, gpt4o, qwen2.5（默认: gpt3.5）",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="模型名称（如果指定，将覆盖 --model-type）",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key（如果不指定则使用默认值）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API Base URL（如果不指定则根据模型类型自动选择）",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="你是一个严格输出 JSON 的多标签分类器。",
        help="系统提示词",
    )
    parser.add_argument(
        "--use-label-desc",
        action="store_true",
        default=False,
        help="是否在提示词中使用标签描述（默认: True）",
    )
    
    # 评估配置
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大测试样本数量，用于调试（默认处理所有样本）",
    )
    
    # 输出配置
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，用于保存预测和指标（绝对路径，默认：根据dataset自动设置）",
    )
    
    # 限流配置
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="每次 API 调用之前的等待时间（秒），用于限流，默认 0 秒",
    )

    # L3R 置信度配置
    parser.add_argument(
        "--l3r-eps",
        type=float,
        default=1e-12,
        help="L3R 计算中的 eps（默认: 1e-12）",
    )
    parser.add_argument(
        "--l3r-alpha",
        type=float,
        default=1.0,
        help="L3R 权重 softmax 的 alpha（默认: 1.0）",
    )
    parser.add_argument(
        "--l3r-agg",
        type=str,
        default="mean",
        choices=["mean", "max", "top-m"],
        help="实例级置信度聚合方式：mean, max, top-m（默认: mean）",
    )
    parser.add_argument(
        "--l3r-top-m",
        type=int,
        default=None,
        help="top-m 聚合中的 m（仅在 l3r-agg=top-m 时有效）",
    )
    parser.add_argument(
        "--l3r-validate",
        action="store_true",
        default=False,
        help="打印一次 L3R 置信度验证信息（默认: False）",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=-1e9,
        help="缓存更新的置信度阈值 tau（默认: -1e9）",
    )

    # SMB 记忆库配置
    parser.add_argument(
        "--bank-type",
        type=str,
        default="naive",
        choices=["naive", "smb"],
        help="记忆库类型：naive 或 smb（默认: naive）",
    )
    parser.add_argument(
        "--smb-B",
        type=int,
        default=100,
        help="SMB 记忆库容量 B（默认: 100）",
    )
    parser.add_argument(
        "--smb-k",
        type=int,
        default=10,
        help="SMB 检索 top-k（默认: 10）",
    )
    parser.add_argument(
        "--smb-lambda1",
        type=float,
        default=1.0,
        help="SMB 覆盖项权重 lambda1（默认: 1.0）",
    )
    parser.add_argument(
        "--smb-lambda2",
        type=float,
        default=1.0,
        help="SMB 多样性项权重 lambda2（默认: 1.0）",
    )
    parser.add_argument(
        "--smb-lambda3",
        type=float,
        default=1.0,
        help="SMB 质量项权重 lambda3（默认: 1.0）",
    )
    parser.add_argument(
        "--smb-epsilon",
        type=float,
        default=1e-12,
        help="SMB 覆盖项 epsilon（默认: 1e-12）",
    )
    parser.add_argument(
        "--smb-W",
        type=int,
        default=200,
        help="SMB candidate_pool 滑动窗口大小 W（默认: 200）",
    )
    
    # 模式配置
    parser.add_argument(
        "--mode",
        type=str,
        default="base",
        choices=["base", "rag"],
        help="实验模式：base（基础模型）或 rag（RAG测试时自适应方法），默认: base",
    )
    
    args = parser.parse_args()
    
    # 根据 dataset 参数自动设置 test_file 和 tag_desc（绝对路径）
    test_file, tag_desc = get_dataset_paths(args.dataset)
    args.test_file = test_file
    args.tag_desc = tag_desc
    
    # 如果未指定 output-dir，则根据 dataset 自动设置（绝对路径）
    if args.output_dir is None:
        dataset_dir = DATASET_NAME_MAP[args.dataset.lower()]
        args.output_dir = os.path.join(PROJECT_ROOT, "rag", "output", dataset_dir)
    
    # 如果未指定 model-name，则根据 model-type 自动设置
    if args.model_name is None:
        args.model_name = get_llm_model_name(args.model_type)
    
    # 如果未指定 base-url，则根据 model-type 自动设置
    if args.base_url is None:
        args.base_url = get_llm_base_url(args.model_type)
    
    # 如果未指定 api-key，则根据 model-type 自动设置
    if args.api_key is None:
        args.api_key = get_llm_api_key(args.model_type)
    
    return args
