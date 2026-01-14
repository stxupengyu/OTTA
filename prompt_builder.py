"""
LLM 提示词构建相关函数。
"""

import json
from typing import Dict, List, Optional


def build_llm_prompt(
    text: str,
    label_to_desc: Dict[str, str],
    use_label_desc: bool = True,
    rag_examples: Optional[List[Dict]] = None,
) -> str:
    """
    构建发送给 LLM 的多标签分类提示词。
    
    Args:
        text: 待分类的文本
        label_to_desc: 标签到描述的映射字典
        use_label_desc: 是否使用标签描述
        rag_examples: 可选的 RAG 示例列表，每个示例包含 "text" 和 "prediction"（包含 labels）
        
    Returns:
        完整的提示词字符串
    """
    # 格式化标签/标签描述列表
    desc_lines = []
    for label, desc in label_to_desc.items():
        if use_label_desc:
            # 将描述中的多个空白字符压缩为单个空格
            short_desc = " ".join(desc.split())
            desc_lines.append(f"- {label}: {short_desc}")
        else:
            # 只使用标签名称，不使用描述
            desc_lines.append(f"- {label}")
    
    labels_str = "\n".join(desc_lines)
    
    # JSON 输出格式示例
    example_output = {
        "labels": list(label_to_desc.keys())[:3]  # 示例：取前3个标签
    }
    
    # 构建 RAG 示例部分（如果提供）
    rag_section = ""
    if rag_examples and len(rag_examples) > 0:
        rag_examples_str = []
        for i, rag_ex in enumerate(rag_examples, 1):
            rag_text = rag_ex.get("text", "")
            rag_pred = rag_ex.get("prediction", {})
            rag_labels = rag_pred.get("labels", []) if isinstance(rag_pred, dict) else []
            rag_examples_str.append(
                f"示例 {i}:\n"
                f"文本: {rag_text}\n"
                f"预测标签: {', '.join(rag_labels)}"
            )
        rag_section = f"""

参考示例（这些是之前高置信度的预测结果，可作为参考）：
{chr(10).join(rag_examples_str)}

"""
    
    prompt = f"""
你是一个多标签分类器。请从下列标签集合中选择所有适用的标签，并输出 JSON 格式：

{labels_str}
{rag_section}
输出要求：
1. 只输出 JSON 格式，不要输出其他内容。
2. JSON 必须包含 "labels" 字段（字符串数组）。
3. 选择所有你认为适用的标签，数量不限。
4. 不要输出 "confidences" 字段。

输出格式示例：
{json.dumps(example_output, ensure_ascii=False, indent=2)}

待分类文本：
{text}
"""
    return prompt.strip()

