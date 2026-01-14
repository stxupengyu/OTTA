"""
Configuration and argument parsing.
"""

import argparse
import os

from models import (
    DEFAULT_MODEL,
    DEFAULT_BASE_URL,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_SILICONFLOW_API_KEY,
)

# Project root directory (resolved at runtime)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset name mapping (argument value -> directory name)
DATASET_NAME_MAP = {
    "movie": "MOVIE",
    "aapd": "AAPD",
    "rcv": "RCV",
    "se": "StackExchange",
}


# LLM model type -> model name
LLM_MODEL_NAME_MAP = {
    "gpt3.5": "gpt-3.5-turbo",
    "gpt4o": "gpt-4o",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
}

# LLM model type -> base URL
LLM_BASE_URL_MAP = {
    "gpt3.5": None,  # None uses OpenAI default
    "gpt4o": None,
    "qwen2.5": "https://api.siliconflow.cn/v1",
}

# LLM model type -> API key (from environment)
LLM_API_KEY_MAP = {
    "gpt3.5": DEFAULT_OPENAI_API_KEY,
    "gpt4o": DEFAULT_OPENAI_API_KEY,
    "qwen2.5": DEFAULT_SILICONFLOW_API_KEY,
}


def get_llm_model_name(model_type: str) -> str:
    """
    Resolve a model name for the given LLM type.
    
    Args:
        model_type: LLM model type (gpt3.5, gpt4o, qwen2.5)
        
    Returns:
        Model name
        
    Raises:
        ValueError: Unsupported model type
    """
    model_type = model_type.lower()
    if model_type not in LLM_MODEL_NAME_MAP:
        raise ValueError(
            f"Unsupported LLM model type: {model_type}. "
            f"Supported types: {list(LLM_MODEL_NAME_MAP.keys())}"
        )
    return LLM_MODEL_NAME_MAP[model_type]


def get_llm_base_url(model_type: str) -> str:
    """
    Resolve a base URL for the given LLM type.
    
    Args:
        model_type: LLM model type
        
    Returns:
        API Base URL (None uses OpenAI default)
    """
    model_type = model_type.lower()
    return LLM_BASE_URL_MAP.get(model_type, None)


def get_llm_api_key(model_type: str) -> str:
    """
    Resolve a default API key for the given LLM type.
    
    Args:
        model_type: LLM model type
        
    Returns:
        API key (may be empty)
    """
    model_type = model_type.lower()
    return LLM_API_KEY_MAP.get(model_type, DEFAULT_OPENAI_API_KEY)


def get_dataset_paths(dataset: str) -> tuple[str, str]:
    """
    Resolve dataset paths (test file + tag descriptions).
    
    Args:
        dataset: Dataset name (movie, aapd, rcv, se)
        
    Returns:
        (test_file_path, tag_desc_path), both absolute paths
        
    Raises:
        ValueError: Unsupported dataset name
    """
    dataset = dataset.lower()
    if dataset not in DATASET_NAME_MAP:
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Supported datasets: {list(DATASET_NAME_MAP.keys())}"
        )
    
    dataset_dir = DATASET_NAME_MAP[dataset]
    data_dir = os.path.join(PROJECT_ROOT, "data", dataset_dir)
    test_file = os.path.join(data_dir, "test.txt")
    tag_desc = os.path.join(data_dir, "tag_description.csv")
    
    return test_file, tag_desc


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="LLM-based multi-label classification evaluation.")
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="movie",
        choices=["movie", "aapd", "rcv", "se"],
        help="Dataset name: movie, aapd, rcv, se (default: movie)",
    )
    
    # Model configuration
    parser.add_argument(
        "--model-type",
        type=str,
        default="gpt3.5",
        choices=["gpt3.5", "gpt4o", "qwen2.5"],
        help="Model type: gpt3.5, gpt4o, qwen2.5 (default: gpt3.5)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name (overrides --model-type when provided)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (defaults from env if omitted)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL (auto-selected by model type if omitted)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a strict multi-label classifier that outputs JSON only.",
        help="System prompt",
    )
    parser.add_argument(
        "--use-label-desc",
        action="store_true",
        default=False,
        help="Use label descriptions in the prompt (default: False)",
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max test samples for quick runs (default: all)",
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for predictions and metrics (default: auto-set per dataset)",
    )
    
    # Rate limit configuration
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="Seconds to wait before each API call (default: 0)",
    )

    # L3R confidence configuration
    parser.add_argument(
        "--l3r-eps",
        type=float,
        default=1e-12,
        help="L3R eps (default: 1e-12)",
    )
    parser.add_argument(
        "--l3r-alpha",
        type=float,
        default=1.0,
        help="L3R softmax alpha (default: 1.0)",
    )
    parser.add_argument(
        "--l3r-agg",
        type=str,
        default="mean",
        choices=["mean", "max", "top-m"],
        help="Instance confidence aggregation: mean, max, top-m (default: mean)",
    )
    parser.add_argument(
        "--l3r-top-m",
        type=int,
        default=None,
        help="m for top-m aggregation (only for l3r-agg=top-m)",
    )
    parser.add_argument(
        "--l3r-validate",
        action="store_true",
        default=False,
        help="Print one L3R validation message (default: False)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=-1e9,
        help="Cache update confidence threshold tau (default: -1e9)",
    )

    # SMB memory bank configuration
    parser.add_argument(
        "--bank-type",
        type=str,
        default="naive",
        choices=["naive", "smb"],
        help="Memory bank type: naive or smb (default: naive)",
    )
    parser.add_argument(
        "--smb-B",
        type=int,
        default=100,
        help="SMB memory size B (default: 100)",
    )
    parser.add_argument(
        "--smb-k",
        type=int,
        default=10,
        help="SMB retrieval top-k (default: 10)",
    )
    parser.add_argument(
        "--smb-lambda1",
        type=float,
        default=1.0,
        help="SMB coverage weight lambda1 (default: 1.0)",
    )
    parser.add_argument(
        "--smb-lambda2",
        type=float,
        default=1.0,
        help="SMB diversity weight lambda2 (default: 1.0)",
    )
    parser.add_argument(
        "--smb-lambda3",
        type=float,
        default=1.0,
        help="SMB quality weight lambda3 (default: 1.0)",
    )
    parser.add_argument(
        "--smb-epsilon",
        type=float,
        default=1e-12,
        help="SMB coverage epsilon (default: 1e-12)",
    )
    parser.add_argument(
        "--smb-W",
        type=int,
        default=200,
        help="SMB candidate_pool sliding window W (default: 200)",
    )
    
    # Mode configuration
    parser.add_argument(
        "--mode",
        type=str,
        default="base",
        choices=["base", "rag"],
        help="Mode: base or rag (default: base)",
    )
    
    args = parser.parse_args()
    
    # Auto-set dataset paths
    test_file, tag_desc = get_dataset_paths(args.dataset)
    args.test_file = test_file
    args.tag_desc = tag_desc
    
    # Auto-set output directory if not provided
    if args.output_dir is None:
        dataset_dir = DATASET_NAME_MAP[args.dataset.lower()]
        args.output_dir = os.path.join(PROJECT_ROOT, "output", dataset_dir)
    
    # Auto-set model name if not provided
    if args.model_name is None:
        args.model_name = get_llm_model_name(args.model_type)
    
    # Auto-set base URL if not provided
    if args.base_url is None:
        args.base_url = get_llm_base_url(args.model_type)
    
    # Auto-set API key if not provided
    if args.api_key is None:
        args.api_key = get_llm_api_key(args.model_type)
    
    return args
