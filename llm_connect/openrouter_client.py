from __future__ import annotations

import requests
import atexit
import json
import time
import random
import logging
import math
import threading
import weakref
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Retryable HTTP status codes
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Default retry configuration
DEFAULT_RETRY_CONFIG = {
    "max_retries": 2,
    "base_delay": 1.5,
    "max_delay": 30.0,
    "jitter": 0.2,
}


def _join_chat_completions_url(base_url: str) -> str:
    trimmed = (base_url or "").strip()
    if not trimmed:
        raise ValueError("base_url must be a non-empty string")
    trimmed = trimmed.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return f"{trimmed}/chat/completions"


class OpenAICompatibleChatClient:
    """
    OpenAI-compatible chat completions client with built-in retry for transient errors.
    Interface: generate(messages, temperature, max_tokens, timeout) -> str
    """

    _session_local = threading.local()
    _session_registry: "weakref.WeakSet[requests.Session]" = weakref.WeakSet()
    _registry_lock = threading.Lock()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
        retry_config: Optional[Dict] = None,
    ):
        self.api_key = api_key or ""
        if not self.api_key:
            raise RuntimeError("API key not provided")
        self.model_name = model_name
        self.base_url = base_url
        self.chat_completions_url = _join_chat_completions_url(base_url)
        self.http_referer = http_referer
        self.x_title = x_title
        self.retry_cfg = {**DEFAULT_RETRY_CONFIG, **(retry_config or {})}

    def _get_session(self) -> requests.Session:
        # Reuse connections per thread to reduce handshake overhead.
        session = getattr(self._session_local, "session", None)
        if session is None:
            session = requests.Session()
            # Conservative pool size to avoid triggering OpenRouter rate limits
            adapter = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=4)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self._session_local.session = session
            self._register_session(session)
        return session

    @classmethod
    def _register_session(cls, session: requests.Session) -> None:
        with cls._registry_lock:
            cls._session_registry.add(session)

    @classmethod
    def _close_registered_sessions(cls) -> None:
        with cls._registry_lock:
            sessions = list(cls._session_registry)
        for session in sessions:
            try:
                session.close()
            except Exception:
                pass

    def _compute_backoff(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Compute backoff delay with exponential increase and jitter."""
        if retry_after is not None:
            return min(retry_after, self.retry_cfg["max_delay"])
        
        base = self.retry_cfg["base_delay"] * (2 ** attempt)
        jitter = self.retry_cfg["jitter"]
        delay = base * (1 + random.uniform(-jitter, jitter))
        return min(delay, self.retry_cfg["max_delay"])

    def _record_usage(self, result: Dict) -> None:
        """Record token usage from API response if tracker is active."""
        from llm_connect.usage_tracker import get_tracker
        
        tracker = get_tracker()
        if tracker is None:
            return
        
        usage = result.get("usage") or {}
        if not usage:
            # No usage data returned - still record the call
            tracker.record_unknown()
            return

        def _to_int(value: object) -> int:
            try:
                if value is None:
                    return 0
                if isinstance(value, bool):
                    return int(value)
                if isinstance(value, int):
                    return value
                if isinstance(value, float):
                    if math.isnan(value) or math.isinf(value):
                        return 0
                    return int(value)
                if isinstance(value, str):
                    raw = value.strip()
                    if not raw:
                        return 0
                    return int(float(raw))
                return int(value)
            except Exception:
                return 0

        # Handle various field names used by different providers
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0

        prompt_val = _to_int(prompt)
        completion_val = _to_int(completion)

        # Fallback: if only total is available, split roughly
        if prompt_val == 0 and completion_val == 0:
            total = _to_int(usage.get("total_tokens"))
            if total > 0:
                prompt_val = total // 2
                completion_val = total - prompt_val

        tracker.record(prompt_val, completion_val)

    def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 20000,
        timeout: int = 60,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        max_retries = self.retry_cfg["max_retries"]
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                session = self._get_session()
                response = session.post(
                    url=self.chat_completions_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=timeout,
                )

                # Check for retryable HTTP status
                if response.status_code in RETRYABLE_STATUS_CODES:
                    retry_after = None
                    if response.status_code == 429:
                        retry_after_header = response.headers.get("Retry-After")
                        if retry_after_header:
                            try:
                                retry_after = float(retry_after_header)
                            except ValueError:
                                pass
                    
                    if attempt < max_retries:
                        delay = self._compute_backoff(attempt, retry_after)
                        logger.debug(
                            f"Retryable HTTP {response.status_code} for {self.model_name}, "
                            f"attempt {attempt + 1}/{max_retries + 1}, sleeping {delay:.1f}s"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise RuntimeError(
                            f"LLM API returned HTTP {response.status_code} after {max_retries + 1} attempts.\n"
                            f"Model: {self.model_name}\n"
                            f"Response: {response.text[:500]}"
                        )

                response.raise_for_status()

                # Parse JSON response
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    if attempt < max_retries:
                        delay = self._compute_backoff(attempt)
                        logger.debug(f"Non-JSON response, retrying in {delay:.1f}s")
                        time.sleep(delay)
                        continue
                    raise RuntimeError(
                        f"LLM API returned non-JSON response.\n"
                        f"Status: {response.status_code}\n"
                        f"Model: {self.model_name}\n"
                        f"Response: {response.text[:500]}"
                    )

                # Validate response structure
                choices = result.get("choices") or []
                if not choices or not choices[0].get("message") or "content" not in choices[0]["message"]:
                    if attempt < max_retries:
                        delay = self._compute_backoff(attempt)
                        logger.debug(f"Empty/invalid response structure, retrying in {delay:.1f}s")
                        time.sleep(delay)
                        continue
                    raise RuntimeError(
                        f"LLM API returned unexpected format.\n"
                        f"Model: {self.model_name}\n"
                        f"Response: {json.dumps(result, indent=2)[:500]}"
                    )

                content = choices[0]["message"]["content"]
                if not content or not content.strip():
                    if attempt < max_retries:
                        delay = self._compute_backoff(attempt)
                        logger.debug(f"Empty content, retrying in {delay:.1f}s")
                        time.sleep(delay)
                        continue
                    finish_reason = choices[0].get("finish_reason", "unknown")
                    raise RuntimeError(
                        f"LLM API returned empty content.\n"
                        f"Model: {self.model_name}\n"
                        f"Finish Reason: {finish_reason}"
                    )

                # Record token usage if tracker is active
                self._record_usage(result)

                return content

            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < max_retries:
                    delay = self._compute_backoff(attempt)
                    logger.debug(f"Timeout for {self.model_name}, retrying in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"LLM API timed out after {timeout}s (tried {max_retries + 1} times).\n"
                    f"Model: {self.model_name}\n"
                    f"Error: {e}"
                ) from e

            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < max_retries:
                    delay = self._compute_backoff(attempt)
                    logger.debug(f"Connection error for {self.model_name}, retrying in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"LLM API connection failed (tried {max_retries + 1} times).\n"
                    f"Model: {self.model_name}\n"
                    f"Error: {e}"
                ) from e

            except requests.exceptions.RequestException as e:
                # Non-retryable request errors (4xx client errors, etc.)
                raise RuntimeError(
                    f"LLM API request failed.\n"
                    f"Model: {self.model_name}\n"
                    f"Error: {e}"
                ) from e

        # Should not reach here, but just in case
        raise RuntimeError(
            f"LLM API failed after {max_retries + 1} attempts.\n"
            f"Model: {self.model_name}\n"
            f"Last Error: {last_error}"
        )


class OpenRouterLLMClient(OpenAICompatibleChatClient):
    """Backward-compatible alias for OpenRouter usage."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
        retry_config: Optional[Dict] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url or "https://openrouter.ai/api/v1",
            http_referer=http_referer,
            x_title=x_title,
            retry_config=retry_config,
        )


atexit.register(OpenAICompatibleChatClient._close_registered_sessions)
