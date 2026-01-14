"""
数据模型和常量定义。
"""

from dataclasses import dataclass
from typing import List


# 默认配置常量
DEFAULT_MODEL = "gpt-3.5-turbo"  # 默认使用的模型名称
DEFAULT_BASE_URL = None  # 默认使用 OpenAI 官方 API（None 表示使用 OpenAI 默认地址）
DEFAULT_OPENAI_API_KEY = "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"  # OpenAI API Key
DEFAULT_SILICONFLOW_API_KEY = "sk-kwloqiyyqcectmrzwzcuhosyabamflhmaclgawpnkvxhabgy"  # SiliconFlow API Key


@dataclass
class Example:
    """表示一个测试样本的数据类。"""
    idx: int  # 样本索引
    text: str  # 待分类的文本内容
    labels: List[str]  # 真实标签列表（ground truth）

