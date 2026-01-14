"""
基于 LLM 的多标签分类器。
"""

import json
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "需要安装 openai 库。请运行: pip install openai"
    )

from l3r_confidence import (
    build_label_prefix_trie,
    build_label_tokenizer,
    compute_l3r_per_label,
)


def _try_fix_json(content: str) -> str:
    """尝试修复常见的 JSON 格式问题。"""
    original = content
    
    # 移除可能的 markdown 代码块标记
    import re
    content = re.sub(r'^```(?:json)?\s*\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'\n```\s*$', '', content, flags=re.MULTILINE)
    content = content.strip()
    
    # 尝试提取 JSON 对象（如果响应中包含额外的文本）
    first_brace = content.find('{')
    if first_brace != -1:
        content = content[first_brace:]
    
    return content if content != original else original


def _parse_json_with_retry(content: str, max_retries: int = 3) -> Dict:
    """解析 JSON，如果失败则尝试修复后重试。"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            last_error = e
            if attempt < max_retries - 1:
                original_content = content
                content = _try_fix_json(content)
                if content == original_content:
                    break
                logging.warning(f"JSON 解析失败（尝试 {attempt + 1}/{max_retries}），尝试修复...")
            else:
                break
    
    logging.error(f"无法解析 JSON: {content[:500]}")
    raise last_error or json.JSONDecodeError("无法解析 JSON", content, 0)


class LLMClassifier:
    """使用大语言模型进行多标签分类的分类器。"""
    
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str],
        api_key: str,
        system_prompt: str,
        request_interval: float = 0.0,
        l3r_eps: float = 1e-12,
        l3r_alpha: float = 1.0,
    ) -> None:
        """
        初始化分类器。
        
        Args:
            model_name: 模型名称
            base_url: API 服务的基础 URL（None 表示使用 OpenAI 默认地址）
            api_key: API 密钥
            system_prompt: 系统提示词
            request_interval: 每次 API 调用之间的等待时间（秒）
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.request_interval = request_interval
        self.l3r_eps = l3r_eps
        self.l3r_alpha = l3r_alpha
        
        # 初始化 OpenAI 客户端
        if not api_key:
            raise RuntimeError("缺少 API Key。")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def predict(
        self,
        prompt: str,
        label_space: Sequence[str],
        max_retries: int = 2,
        return_logprobs: bool = False,
    ) -> Dict:
        """
        对给定提示词进行分类预测。
        
        Args:
            prompt: 分类提示词
            label_space: 标签空间（所有可能的标签列表）
            max_retries: API 调用失败时的最大重试次数
            return_logprobs: 是否返回 logprobs 和置信度
        
        Returns:
            如果 return_logprobs=False，返回预测的标签列表
            如果 return_logprobs=True，返回包含 "labels" 和 "confidences" 的字典
        """
        last_error = None
        
        # 重试机制
        for attempt in range(max_retries + 1):
            try:
                # 在调用 API 之前等待指定时间（用于限流）
                if self.request_interval > 0:
                    time.sleep(self.request_interval)
                
                # 调用 API 进行预测
                create_kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,  # 温度设为 0 以获得确定性输出
                    "response_format": {"type": "json_object"},  # 强制 JSON 格式
                }
                
                # 如果需要返回logprobs，添加logprobs参数
                if return_logprobs:
                    create_kwargs["logprobs"] = True
                    try:
                        create_kwargs["top_logprobs"] = 5  # 返回 top-5 的 logprobs
                    except Exception:
                        pass
                
                response = self._client.chat.completions.create(**create_kwargs)
                content = response.choices[0].message.content
                
                # 解析 JSON
                result = _parse_json_with_retry(content, max_retries=3)
                
                # 提取标签列表
                labels = result.get("labels", [])
                
                # 验证标签是否在标签空间中
                valid_labels = [label for label in labels if label in label_space]
                
                if not valid_labels:
                    logging.warning(f"预测结果中没有有效的标签。原始输出: {labels}")
                
                # 如果不需要 logprobs，直接返回标签列表（保持向后兼容）
                if not return_logprobs:
                    return valid_labels
                
                # 如果需要 logprobs，计算置信度
                result_dict = {"labels": valid_labels}
                
                if return_logprobs:
                    # 提取 logprobs
                    logprobs_obj = getattr(response.choices[0], 'logprobs', None)
                    if logprobs_obj:
                        token_logprobs = self._extract_token_logprobs(logprobs_obj)
                        confidences = self._compute_label_confidences_from_logprobs(
                            valid_labels, content, token_logprobs, label_space
                        )
                        result_dict["confidences"] = confidences
                    else:
                        logging.warning("API 响应中没有 logprobs 信息")
                        result_dict["confidences"] = {}
                
                return result_dict
                
            except json.JSONDecodeError as e:
                last_error = e
                if attempt < max_retries:
                    logging.warning(f"JSON 解析失败（尝试 {attempt + 1}/{max_retries + 1}），将重试...")
                    time.sleep(1.0)
                else:
                    logging.error(f"JSON 解析失败，已达到最大重试次数。")
                    if return_logprobs:
                        return {"labels": [], "confidences": {}}
                    return []
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logging.warning(f"API 调用失败（尝试 {attempt + 1}/{max_retries + 1}）: {e}，将重试...")
                    time.sleep(1.0)
                else:
                    raise
        
        # 不应该到达这里
        raise RuntimeError(f"分类失败: {last_error}")
    
    def _extract_token_logprobs(self, logprobs_obj) -> List[Dict]:
        """
        从API响应中提取token logprobs信息。
        
        Args:
            logprobs_obj: API返回的logprobs对象
            
        Returns:
            List of per-step token logprob data with optional top_logprobs
        """
        token_logprobs: List[Dict] = []
        
        # 尝试不同的方式提取 logprobs
        if hasattr(logprobs_obj, 'content') and logprobs_obj.content:
            for token_info in logprobs_obj.content:
                token = token_info.token if hasattr(token_info, "token") else None
                logprob = token_info.logprob if hasattr(token_info, "logprob") else None
                top_logprobs = {}
                if hasattr(token_info, "top_logprobs") and token_info.top_logprobs:
                    for top_info in token_info.top_logprobs:
                        top_token = top_info.token if hasattr(top_info, "token") else None
                        top_lp = top_info.logprob if hasattr(top_info, "logprob") else None
                        if top_token is not None and top_lp is not None:
                            top_logprobs[top_token] = top_lp
                if logprob is not None:
                    token_logprobs.append(
                        {
                            "token": token,
                            "logprob": logprob,
                            "top_logprobs": top_logprobs,
                        }
                    )
        elif hasattr(logprobs_obj, 'tokens') and logprobs_obj.tokens:
            for token_info in logprobs_obj.tokens:
                token = token_info.token if hasattr(token_info, "token") else None
                logprob = token_info.logprob if hasattr(token_info, "logprob") else None
                top_logprobs = {}
                if hasattr(token_info, "top_logprobs") and token_info.top_logprobs:
                    for top_info in token_info.top_logprobs:
                        top_token = top_info.token if hasattr(top_info, "token") else None
                        top_lp = top_info.logprob if hasattr(top_info, "logprob") else None
                        if top_token is not None and top_lp is not None:
                            top_logprobs[top_token] = top_lp
                if logprob is not None:
                    token_logprobs.append(
                        {
                            "token": token,
                            "logprob": logprob,
                            "top_logprobs": top_logprobs,
                        }
                    )
        
        return token_logprobs
    
    def _compute_label_confidences_from_logprobs(
        self, 
        labels: List[str], 
        content: str, 
        token_logprobs: List[Dict],
        label_space: Sequence[str],
    ) -> Dict[str, float]:
        """
        基于 L3R 计算每个标签的置信度。
        
        Args:
            labels: 预测的标签列表
            content: API返回的原始内容
            token_logprobs: 每步 token 的 logprob 与 top_logprobs
            label_space: 标签空间
        
        Returns:
            标签到置信度的字典
        """
        if not token_logprobs or not labels:
            return {}

        tokenizer = build_label_tokenizer(token_logprobs)
        trie = build_label_prefix_trie(label_space, tokenizer)
        return compute_l3r_per_label(
            predicted_labels=labels,
            label_space=label_space,
            token_logprobs=token_logprobs,
            content=content,
            trie=trie,
            eps=self.l3r_eps,
            alpha=self.l3r_alpha,
            tokenizer=tokenizer,
        )
