"""
工具函数，包含 embedding 相关功能。
"""

import logging
from typing import List, Tuple

# 全局变量：用于缓存 embedding 模型
_embedding_model = None


def _get_embedding_model():
    """
    获取或初始化 embedding 模型（使用单例模式，避免重复加载）。
    
    Returns:
        sentence_transformers 模型实例
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logging.info("正在加载 embedding 模型: sentence-transformers/all-mpnet-base-v2")
            _embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            logging.info("Embedding 模型加载完成")
        except ImportError:
            raise ImportError(
                "需要安装 sentence-transformers 库。请运行: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"加载 embedding 模型失败: {e}")
    return _embedding_model


def get_text_embedding(text: str) -> List[float]:
    """
    获取文本的 embedding。
    
    Args:
        text: 输入文本
        
    Returns:
        embedding 向量（列表）
    """
    if not text:
        return []
    
    try:
        model = _get_embedding_model()
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        logging.error(f"计算 embedding 时出错: {e}")
        return []


def find_top_similar_texts(
    query_text: str, 
    cache_texts: List[str], 
    cache_embeddings: List[List[float]],
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    在cache中找到与query_text最相似的top-k个文本。
    
    使用 embedding 相似度进行检索，批量计算以提高效率。
    
    Args:
        query_text: 查询文本
        cache_texts: cache中的文本列表
        cache_embeddings: cache中对应的embedding列表
        top_k: 返回top-k个最相似的文本
    
    Returns:
        List of (index, similarity_score) tuples，按相似度降序排列
    """
    if not query_text or not cache_texts or not cache_embeddings:
        return []
    
    if len(cache_texts) != len(cache_embeddings):
        logging.warning(f"cache_texts 和 cache_embeddings 长度不匹配: {len(cache_texts)} vs {len(cache_embeddings)}")
        return []
    
    try:
        import numpy as np
        model = _get_embedding_model()
        
        # 获取查询文本的 embedding
        query_embedding = model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
        
        # 将 cache_embeddings 转换为 numpy 数组
        cache_embeddings_array = np.array(cache_embeddings)
        
        # 批量计算相似度（dot product，因为 embeddings 已归一化）
        similarities = query_embedding.dot(cache_embeddings_array.T)
        
        # 归一化到 0-1 范围（dot product 的范围通常是 -1 到 1）
        similarities = (similarities + 1.0) / 2.0
        similarities = np.clip(similarities, 0.0, 1.0)
        
        # 获取 top-k 索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 构建结果列表
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results
        
    except Exception as e:
        logging.error(f"使用 embedding 检索相似文本时出错: {e}")
        return []

