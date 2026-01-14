"""
Utility helpers for embeddings.
"""

import logging
from typing import List, Tuple

# Global cache for the embedding model
_embedding_model = None


def _get_embedding_model():
    """
    Get or initialize the embedding model (singleton).

    Returns:
        sentence_transformers model instance
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logging.info("Loading embedding model: sentence-transformers/all-mpnet-base-v2")
            _embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            logging.info("Embedding model loaded")
        except ImportError:
            raise ImportError(
                "Missing dependency: sentence-transformers. Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
    return _embedding_model


def get_text_embedding(text: str) -> List[float]:
    """
    Compute an embedding for the input text.

    Args:
        text: Input text

    Returns:
        Embedding vector as a list
    """
    if not text:
        return []
    
    try:
        model = _get_embedding_model()
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        logging.error(f"Embedding computation failed: {e}")
        return []


def find_top_similar_texts(
    query_text: str, 
    cache_texts: List[str], 
    cache_embeddings: List[List[float]],
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Find top-k most similar cached texts to the query text.

    Similarity uses normalized embedding dot product.

    Args:
        query_text: Query text
        cache_texts: Cached texts
        cache_embeddings: Cached embeddings
        top_k: Number of results

    Returns:
        List of (index, similarity_score) sorted by similarity desc
    """
    if not query_text or not cache_texts or not cache_embeddings:
        return []
    
    if len(cache_texts) != len(cache_embeddings):
        logging.warning(
            "cache_texts and cache_embeddings length mismatch: %s vs %s",
            len(cache_texts),
            len(cache_embeddings),
        )
        return []
    
    try:
        import numpy as np
        model = _get_embedding_model()
        
        # Encode query text
        query_embedding = model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
        
        # Convert cached embeddings
        cache_embeddings_array = np.array(cache_embeddings)
        
        # Batch similarity via dot product (embeddings are normalized)
        similarities = query_embedding.dot(cache_embeddings_array.T)
        
        # Normalize to [0, 1]
        similarities = (similarities + 1.0) / 2.0
        similarities = np.clip(similarities, 0.0, 1.0)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build result list
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results
        
    except Exception as e:
        logging.error(f"Embedding-based retrieval failed: {e}")
        return []
