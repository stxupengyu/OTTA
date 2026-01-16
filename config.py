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

# Project root directory (absolute path)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Dataset name mapping (arg value -> directory name)
DATASET_NAME_MAP = {
    "movie": "MOVIE",
    "aapd": "AAPD",
    "rcv": "RCV",
    "se": "StackExchange",
}

# Dataset description mapping
DATASET_DESC_MAP = {
    "movie": (
        "This is a movie genre classification task. Based on plot summaries or descriptions, "
        "assign one or more genre labels (for example: action, comedy, sci-fi, thriller, romance). "
        "A movie can belong to multiple genres. Identify key information such as setting, plot "
        "elements, and tone. Genre labels may overlap, so consider the overall characteristics."
    ),
    "aapd": (
        "This is an academic paper topic classification task. Based on CS paper abstracts, "
        "identify one or more research topics or fields. Abstracts typically include methods, "
        "technical contributions, and application domains. A paper can span multiple areas, so "
        "use technical terms and application context to make accurate decisions."
    ),
    "rcv": (
        "This is a news topic classification task. Based on Reuters news articles, identify one "
        "or more topics, industries, and regional labels. Articles may include headline, summary, "
        "and body text. A single article can involve multiple topics, industries, or regions, so "
        "analyze key entities, events, and background information."
    ),
    "se": (
        "This is a technical Q&A tag recommendation task. Based on questions from StackExchange-like "
        "communities (title and body), assign one or more technical tags. Questions may include code, "
        "errors, and technical terms. A question can have multiple tags, but no more than 5. Focus on "
        "languages, frameworks, tools, and the technical context."
    ),
}


def get_dataset_desc(dataset: str) -> str:
    """
    Get the dataset description by name.

    Args:
        dataset: Dataset name (movie, aapd, rcv, se)

    Returns:
        Dataset description string

    Raises:
        ValueError: If dataset name is not supported
    """
    dataset = dataset.lower()
    if dataset not in DATASET_DESC_MAP:
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Supported datasets: {list(DATASET_DESC_MAP.keys())}"
        )
    return DATASET_DESC_MAP[dataset]


# LLM model type to model name mapping
LLM_MODEL_NAME_MAP = {
    "gpt3.5": "gpt-3.5-turbo",
    "gpt4o": "gpt-4o",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
}

# LLM model type to API base URL mapping
LLM_BASE_URL_MAP = {
    "gpt3.5": None,  # None uses OpenAI default base URL
    "gpt4o": None,
    "qwen2.5": "https://api.siliconflow.cn/v1",
}

# LLM model type to API key mapping
LLM_API_KEY_MAP = {
    "gpt3.5": DEFAULT_OPENAI_API_KEY,
    "gpt4o": DEFAULT_OPENAI_API_KEY,
    "qwen2.5": DEFAULT_SILICONFLOW_API_KEY,
}

ENV_API_KEY_MAP = {
    "gpt3.5": "OPENAI_API_KEY",
    "gpt4o": "OPENAI_API_KEY",
    "qwen2.5": "SILICONFLOW_API_KEY",
}


def get_llm_model_name(model_type: str) -> str:
    """
    Get model name by LLM model type.

    Args:
        model_type: LLM model type (gpt3.5, gpt4o, qwen2.5)

    Returns:
        Model name

    Raises:
        ValueError: If model type is not supported
    """
    model_type = model_type.lower()
    if model_type not in LLM_MODEL_NAME_MAP:
        raise ValueError(
            f"Unsupported LLM model type: {model_type}. "
            f"Supported model types: {list(LLM_MODEL_NAME_MAP.keys())}"
        )
    return LLM_MODEL_NAME_MAP[model_type]


def get_llm_base_url(model_type: str) -> str:
    """
    Get API base URL by LLM model type.

    Args:
        model_type: LLM model type

    Returns:
        API base URL (None uses OpenAI default base URL)
    """
    model_type = model_type.lower()
    return LLM_BASE_URL_MAP.get(model_type, None)


def get_llm_api_key(model_type: str) -> str:
    """
    Get default API key for a model type.

    Args:
        model_type: LLM model type

    Returns:
        API key string (empty if not configured)
    """
    model_type = model_type.lower()
    env_key = ENV_API_KEY_MAP.get(model_type, "OPENAI_API_KEY")
    env_value = os.getenv(env_key, "").strip()
    if env_value:
        return env_value
    return LLM_API_KEY_MAP.get(model_type, DEFAULT_OPENAI_API_KEY)


def get_dataset_paths(dataset: str) -> tuple[str, str]:
    """
    Get test file and tag description file paths by dataset name.

    Args:
        dataset: Dataset name (movie, aapd, rcv, se)

    Returns:
        (test_file_path, tag_desc_path) tuple with absolute paths

    Raises:
        ValueError: If dataset name is not supported
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
    Parse command-line arguments.

    Returns:
        Namespace with parsed arguments
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
        help="Model name (overrides --model-type when set)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (uses env var if not provided)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL (auto-selected by model type when unset)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a multi-label classifier that must output strict JSON.",
        help="System prompt",
    )
    parser.add_argument(
        "--use-label-desc",
        action="store_true",
        default=False,
        help="Include label descriptions in the prompt (default: False)",
    )
    parser.add_argument(
        "--use-dataset-desc",
        action="store_true",
        default=False,
        help="Include dataset description in the prompt (default: False)",
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max test samples for debugging (default: all samples)",
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for predictions/metrics (absolute path, default: set by dataset)",
    )
    
    # Rate limiting configuration
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="Sleep time before each API call in seconds (default: 0)",
    )

    # L3R confidence configuration
    parser.add_argument(
        "--conf-type",
        type=str,
        default="l3r",
        choices=["naive", "l3r"],
        help="Confidence type: naive or l3r (default: l3r)",
    )
    parser.add_argument(
        "--l3r-eps",
        type=float,
        default=1e-12,
        help="Epsilon for L3R computation (default: 1e-12)",
    )
    parser.add_argument(
        "--l3r-alpha",
        type=float,
        default=1.0,
        help="Alpha for L3R softmax weighting (default: 1.0)",
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
        help="m for top-m aggregation (only when l3r-agg=top-m)",
    )
    parser.add_argument(
        "--l3r-validate",
        action="store_true",
        default=False,
        help="Print one-time L3R validation info (default: False)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=-1e9,
        help="Confidence threshold tau for cache update (default: -1e9)",
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
        "--cache-size",
        type=int,
        default=100,
        help="Memory bank capacity (naive and smb, default: 100)",
    )
    parser.add_argument(
        "--rag-k",
        type=int,
        default=10,
        help="Top-k retrieval for RAG (naive and smb, default: 10)",
    )
    parser.add_argument(
        "--smb-B",
        type=int,
        default=None,
        help="SMB capacity B (deprecated, use --cache-size)",
    )
    parser.add_argument(
        "--smb-k",
        type=int,
        default=None,
        help="SMB top-k (deprecated, use --rag-k)",
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
        help="SMB candidate_pool sliding window size W (default: 200)",
    )
    
    # RAG mode configuration
    parser.add_argument(
        "--rag-warmup",
        type=int,
        default=200,
        help="Warmup samples before RAG retrieval starts (default: 200)",
    )
    parser.add_argument(
        "--encoder-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Encoder model for text embeddings (default: sentence-transformers/all-mpnet-base-v2)",
    )
    parser.add_argument(
        "--se-candidate-topk",
        type=int,
        default=100,
        help="se dataset coarse stage candidate label count (default: 100)",
    )
    parser.add_argument(
        "--se-label-embed-batch-size",
        type=int,
        default=256,
        help="se dataset label embedding batch size (default: 256)",
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
    
    # Auto-set test_file and tag_desc based on dataset
    test_file, tag_desc = get_dataset_paths(args.dataset)
    args.test_file = test_file
    args.tag_desc = tag_desc
    
    # Auto-set output_dir based on dataset
    if args.output_dir is None:
        dataset_dir = DATASET_NAME_MAP[args.dataset.lower()]
        args.output_dir = os.path.join(PROJECT_ROOT, "output", dataset_dir)
    
    # Auto-set model name
    if args.model_name is None:
        args.model_name = get_llm_model_name(args.model_type)
    
    # Auto-set base URL
    if args.base_url is None:
        args.base_url = get_llm_base_url(args.model_type)
    
    # Auto-set API key
    if args.api_key is None:
        args.api_key = get_llm_api_key(args.model_type)
    
    # Backward compatibility: --smb-B overrides --cache-size
    if args.smb_B is not None:
        args.cache_size = args.smb_B
    
    return args
