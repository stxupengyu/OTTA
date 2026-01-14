"""
LLM-based multi-label classifier.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "Missing dependency: openai. Install with: pip install openai"
    )

from l3r_confidence import (
    build_label_prefix_trie,
    build_label_tokenizer,
    compute_l3r_per_label,
)


def _try_fix_json(content: str) -> str:
    """Try to fix common JSON formatting issues."""
    original = content
    
    # Strip markdown code fences
    import re
    content = re.sub(r'^```(?:json)?\s*\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'\n```\s*$', '', content, flags=re.MULTILINE)
    content = content.strip()
    
    # Try to extract a JSON object from extra text
    first_brace = content.find('{')
    if first_brace != -1:
        content = content[first_brace:]
    
    return content if content != original else original


def _parse_json_with_retry(content: str, max_retries: int = 3) -> Dict:
    """Parse JSON with optional fix-and-retry."""
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
                logging.warning(
                    "JSON parsing failed (attempt %s/%s). Trying to fix.",
                    attempt + 1,
                    max_retries,
                )
            else:
                break
    
    logging.error("Failed to parse JSON: %s", content[:500])
    raise last_error or json.JSONDecodeError("Failed to parse JSON", content, 0)


class LLMClassifier:
    """Multi-label classifier powered by an LLM."""
    
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
        Initialize the classifier.

        Args:
            model_name: Model name
            base_url: API base URL (None uses OpenAI default)
            api_key: API key
            system_prompt: System prompt
            request_interval: Seconds to wait between API calls
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.request_interval = request_interval
        self.l3r_eps = l3r_eps
        self.l3r_alpha = l3r_alpha
        
        # Initialize OpenAI client
        if not api_key:
            raise RuntimeError("Missing API key.")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def predict(
        self,
        prompt: str,
        label_space: Sequence[str],
        max_retries: int = 2,
        return_logprobs: bool = False,
    ) -> Dict:
        """
        Run a classification prediction for the prompt.

        Args:
            prompt: Classification prompt
            label_space: Full label space
            max_retries: Max retries on API failure
            return_logprobs: Whether to return logprobs/confidences

        Returns:
            If return_logprobs=False, returns list of labels
            If return_logprobs=True, returns dict with "labels" and "confidences"
        """
        last_error = None
        
        # Retry loop
        for attempt in range(max_retries + 1):
            try:
                # Optional delay before API call
                if self.request_interval > 0:
                    time.sleep(self.request_interval)
                
                # Call API
                create_kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,  # Deterministic outputs
                    "response_format": {"type": "json_object"},  # Force JSON format
                }
                
                # Add logprobs if requested
                if return_logprobs:
                    create_kwargs["logprobs"] = True
                    try:
                        create_kwargs["top_logprobs"] = 5  # top-5 logprobs
                    except Exception:
                        pass
                
                response = self._client.chat.completions.create(**create_kwargs)
                content = response.choices[0].message.content
                
                # Parse JSON
                result = _parse_json_with_retry(content, max_retries=3)
                
                # Extract labels
                labels = result.get("labels", [])
                
                # Filter labels to the known label space
                valid_labels = [label for label in labels if label in label_space]
                
                if not valid_labels:
                    logging.warning("No valid labels in output. Raw: %s", labels)
                
                # Return labels only if logprobs not requested
                if not return_logprobs:
                    return valid_labels
                
                # Compute confidences from logprobs
                result_dict = {"labels": valid_labels}
                
                if return_logprobs:
                    # Extract logprobs
                    logprobs_obj = getattr(response.choices[0], 'logprobs', None)
                    if logprobs_obj:
                        token_logprobs = self._extract_token_logprobs(logprobs_obj)
                        confidences = self._compute_label_confidences_from_logprobs(
                            valid_labels, content, token_logprobs, label_space
                        )
                        result_dict["confidences"] = confidences
                    else:
                        logging.warning("API response has no logprobs")
                        result_dict["confidences"] = {}
                
                return result_dict
                
            except json.JSONDecodeError as e:
                last_error = e
                if attempt < max_retries:
                    logging.warning(
                        "JSON parsing failed (attempt %s/%s). Retrying...",
                        attempt + 1,
                        max_retries + 1,
                    )
                    time.sleep(1.0)
                else:
                    logging.error("JSON parsing failed after max retries.")
                    if return_logprobs:
                        return {"labels": [], "confidences": {}}
                    return []
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logging.warning(
                        "API call failed (attempt %s/%s): %s. Retrying...",
                        attempt + 1,
                        max_retries + 1,
                        e,
                    )
                    time.sleep(1.0)
                else:
                    raise
        
        # Should not reach here
        raise RuntimeError(f"Classification failed: {last_error}")
    
    def _extract_token_logprobs(self, logprobs_obj) -> List[Dict]:
        """
        Extract per-token logprobs from the API response.

        Args:
            logprobs_obj: API logprobs object

        Returns:
            List of per-step token logprob data with optional top_logprobs
        """
        token_logprobs: List[Dict] = []
        
        # Handle different logprobs formats
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
        Compute per-label confidence using L3R.

        Args:
            labels: Predicted labels
            content: Raw API content
            token_logprobs: Per-step token logprobs with top_logprobs
            label_space: Full label space

        Returns:
            Dict mapping label -> confidence
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
