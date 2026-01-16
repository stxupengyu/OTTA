"""
Utility functions for embeddings.
"""

import logging
from typing import List, Tuple

# Global cache for embedding model and model name
_embedding_model = None
_encoder_model_name = None


def init_embedding_model(model_name: str) -> None:
    """
    Initialize the embedding model name.

    Args:
        model_name: Model name (e.g., sentence-transformers/all-mpnet-base-v2)
    """
    global _encoder_model_name, _embedding_model
    # Reload model if the name changes
    if _encoder_model_name != model_name:
        _encoder_model_name = model_name
        _embedding_model = None  # Reset to force reload


def _get_embedding_model():
    """
    Get or initialize the embedding model (singleton).

    Returns:
        sentence_transformers model instance
    """
    global _embedding_model, _encoder_model_name
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use configured model name or default
            model_name = _encoder_model_name or "sentence-transformers/all-mpnet-base-v2"
            logging.info(f"Loading embedding model: {model_name}")
            _embedding_model = SentenceTransformer(model_name)
            logging.info("Embedding model loaded")
        except ImportError:
            raise ImportError(
                "The sentence-transformers library is required. Install it with: "
                "pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
    return _embedding_model


def get_text_embedding(text: str) -> List[float]:
    """
    Get embedding for a single text.

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
        logging.error(f"Error computing embedding: {e}")
        return []


def get_text_embeddings(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Get embeddings for a batch of texts.

    Args:
        texts: Input text list
        batch_size: Batch size for encoding

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    try:
        model = _get_embedding_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
    except Exception as e:
        logging.error(f"Error computing batch embeddings: {e}")
        return []


def find_top_similar_texts(
    query_text: str, 
    cache_texts: List[str], 
    cache_embeddings: List[List[float]],
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Find top-k most similar texts in the cache.

    Uses embedding similarity for retrieval.

    Args:
        query_text: Query text
        cache_texts: Cached text list
        cache_embeddings: Cached embeddings list
        top_k: Number of top results

    Returns:
        List of (index, similarity_score) tuples sorted by similarity desc
    """
    if not query_text or not cache_texts or not cache_embeddings:
        return []
    
    if len(cache_texts) != len(cache_embeddings):
        logging.warning(
            f"cache_texts and cache_embeddings length mismatch: "
            f"{len(cache_texts)} vs {len(cache_embeddings)}"
        )
        return []
    
    try:
        import numpy as np
        model = _get_embedding_model()

        # Embed query text
        query_embedding = model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)

        # Convert cached embeddings to array
        cache_embeddings_array = np.array(cache_embeddings)

        # Batch similarity (dot product; embeddings are normalized)
        similarities = query_embedding.dot(cache_embeddings_array.T)

        # Normalize to 0-1 range (dot product is usually -1 to 1)
        similarities = (similarities + 1.0) / 2.0
        similarities = np.clip(similarities, 0.0, 1.0)

        # Top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]

        return results

    except Exception as e:
        logging.error(f"Error retrieving similar texts via embeddings: {e}")
        return []
