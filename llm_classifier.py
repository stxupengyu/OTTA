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
        "The openai library is required. Install it with: pip install openai"
    )

from l3r_confidence import (
    build_label_prefix_trie,
    build_label_tokenizer,
    compute_l3r_per_label,
    compute_naive_per_label,
)


def _try_fix_json(content: str) -> str:
    """Try to fix common JSON formatting issues."""
    original = content

    # Strip possible markdown code fences
    import re
    content = re.sub(r'^```(?:json)?\s*\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'\n```\s*$', '', content, flags=re.MULTILINE)
    content = content.strip()

    # Try to extract JSON object if extra text exists
    first_brace = content.find('{')
    if first_brace != -1:
        content = content[first_brace:]

    return content if content != original else original


def _parse_json_with_retry(content: str, max_retries: int = 3) -> Dict:
    """Parse JSON with retry and best-effort fixes."""
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
                    f"JSON parse failed (attempt {attempt + 1}/{max_retries}); trying to fix..."
                )
            else:
                break

    logging.error(f"Failed to parse JSON: {content[:500]}")
    raise last_error or json.JSONDecodeError("Failed to parse JSON", content, 0)


class LLMClassifier:
    """Multi-label classifier using an LLM."""
    
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str],
        api_key: str,
        system_prompt: str,
        request_interval: float = 0.0,
        l3r_eps: float = 1e-12,
        l3r_alpha: float = 1.0,
        conf_type: str = "l3r",
    ) -> None:
        """
        Initialize the classifier.

        Args:
            model_name: Model name
            base_url: API base URL (None uses OpenAI default)
            api_key: API key
            system_prompt: System prompt
            request_interval: Sleep time between API calls (seconds)
            conf_type: Confidence type (naive or l3r)
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.request_interval = request_interval
        self.l3r_eps = l3r_eps
        self.l3r_alpha = l3r_alpha
        self.conf_type = conf_type
        
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
        Run classification for a prompt.

        Args:
            prompt: Classification prompt
            label_space: All possible labels
            max_retries: Max retries on API failure
            return_logprobs: Whether to return logprobs and confidences

        Returns:
            If return_logprobs=False, returns list of predicted labels.
            If return_logprobs=True, returns dict with "labels" and "confidences".
        """
        last_error = None

        # Retry loop
        for attempt in range(max_retries + 1):
            try:
                # Wait before calling API (rate limiting)
                if self.request_interval > 0:
                    time.sleep(self.request_interval)

                # Call API for prediction
                create_kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,  # Deterministic output
                    "response_format": {"type": "json_object"},  # Enforce JSON output
                }

                # Add logprobs parameters when requested
                if return_logprobs:
                    create_kwargs["logprobs"] = True
                    try:
                        create_kwargs["top_logprobs"] = 5  # Top-5 logprobs
                    except Exception:
                        pass

                response = self._client.chat.completions.create(**create_kwargs)
                content = response.choices[0].message.content

                # Parse JSON
                result = _parse_json_with_retry(content, max_retries=3)

                # Extract labels
                labels = result.get("labels", [])

                # Validate labels
                valid_labels = [label for label in labels if label in label_space]

                if not valid_labels:
                    logging.warning(
                        f"No valid labels in prediction. Raw output: {labels}"
                    )

                # Return labels if no logprobs requested
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
                        logging.warning("API response missing logprobs data")
                        result_dict["confidences"] = {}

                return result_dict

            except json.JSONDecodeError as e:
                last_error = e
                if attempt < max_retries:
                    logging.warning(
                        f"JSON parse failed (attempt {attempt + 1}/{max_retries + 1}); retrying..."
                    )
                    time.sleep(1.0)
                else:
                    logging.error("JSON parse failed; max retries reached.")
                    if return_logprobs:
                        return {"labels": [], "confidences": {}}
                    return []
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logging.warning(
                        f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying..."
                    )
                    time.sleep(1.0)
                else:
                    raise

        # Should not reach here
        raise RuntimeError(f"Classification failed: {last_error}")

    def _extract_token_logprobs(self, logprobs_obj) -> List[Dict]:
        """
        Extract token logprobs from API response.

        Args:
            logprobs_obj: API logprobs object

        Returns:
            List of per-step token logprob data with optional top_logprobs
        """
        token_logprobs: List[Dict] = []

        # Try multiple logprobs formats
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
        Compute per-label confidence (L3R or naive).

        Args:
            labels: Predicted labels
            content: Raw API response content
            token_logprobs: Token logprobs with optional top_logprobs
            label_space: Label space

        Returns:
            Mapping from label to confidence
        """
        if not token_logprobs or not labels:
            return {}

        if self.conf_type == "naive":
            return compute_naive_per_label(
                predicted_labels=labels,
                token_logprobs=token_logprobs,
                content=content,
            )

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
