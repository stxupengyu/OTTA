"""
数据加载相关函数。
"""

import csv
from typing import Dict, List

from models import Example


def load_tag_descriptions(path: str) -> Dict[str, str]:
    """
    从 CSV 文件中加载标签及其描述信息。
    
    Args:
        path: CSV 文件路径，应包含 "tags" 和 "outputs" 两列
        
    Returns:
        标签到描述的字典映射，键为标签名，值为标签描述
    """
    label_to_desc: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = (row.get("tags") or "").strip()
            if not tag:
                continue
            desc = (row.get("outputs") or "").strip()
            # 统一换行符格式
            label_to_desc[tag] = desc.replace("\r\n", "\n").strip()
    return label_to_desc


def load_test_file(path: str) -> List[Example]:
    """
    从测试文件中加载样本数据。
    
    测试文件格式：
        Text: <文本内容>
        Labels: <标签1>,<标签2>,...
        （空行分隔不同样本）
    
    Args:
        path: 测试文件路径
        
    Returns:
        样本列表，每个样本包含索引、文本和标签
    """
    examples: List[Example] = []
    current_text = None  # 当前正在读取的文本
    idx = 0  # 样本计数器

    def flush(labels_line: str) -> None:
        """将当前文本和标签组合成一个样本并添加到列表。"""
        nonlocal idx, current_text
        if current_text is None:
            return
        # 解析标签（逗号分隔）
        labels = [lab.strip() for lab in labels_line.split(",") if lab.strip()]
        idx += 1
        examples.append(Example(idx=idx, text=current_text.strip(), labels=labels))
        current_text = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 解析 "Text:" 开头的行
            if line.startswith("Text:"):
                current_text = line[len("Text:"):].strip()
            # 解析 "Labels:" 开头的行，并完成一个样本的读取
            elif line.startswith("Labels:"):
                flush(line[len("Labels:"):].strip())

    return examples

