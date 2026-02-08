# kabb/llm_client.py
import os
import asyncio
import logging
from typing import List, Union, Optional, Any, Dict

logging.basicConfig(level=logging.INFO)

class LLMClient:
    """
    Wrapper for interacting with LLM providers.
    Supports:
      - together (original)
      - openrouter (OpenAI-compatible API)
    """

    def __init__(
        self,
        api_key: Union[str, List[str]],
        provider: str = "together",
        max_retries: int = 4,
        retry_backoff: List[int] = [1, 2, 4, 8],
        timeout: int = 60,
        # Optional headers for OpenRouter (recommended but not strictly required)
        openrouter_headers: Optional[Dict[str, str]] = None,
    ):
        if api_key is None or api_key == "":
            # Keep backward compatibility
            api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

        if isinstance(api_key, str):
            self.api_key = [api_key]
        elif isinstance(api_key, list) and all(isinstance(key, str) for key in api_key):
            self.api_key = api_key
        else:
            raise ValueError("api_key must be a string or a list of strings.")

        self.current_key_index = 0
        self.provider = (provider or "together").lower()
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout = timeout

        self.openrouter_headers = openrouter_headers or {
            # You can set these in env and pass them in if you want; safe defaults:
            # OpenRouter recommends setting HTTP-Referer and X-Title for analytics/rate-limits.
            # Leaving them blank is usually OK, but some org setups prefer these.
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", ""),
            "X-Title": os.environ.get("OPENROUTER_X_TITLE", "kabb"),
        }

        self._init_client()

    def _init_client(self):
        if self.provider == "together":
            # Keep original behavior
            import together
            from together import AsyncTogether
            self._together_module = together
            self.client = AsyncTogether(api_key=self.api_key[self.current_key_index])

        elif self.provider == "openrouter":
            # OpenRouter is OpenAI-compatible
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=self.api_key[self.current_key_index],
                base_url="https://openrouter.ai/api/v1",
                default_headers={k: v for k, v in self.openrouter_headers.items() if v},
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def switch_api_key(self) -> bool:
        if self.current_key_index < len(self.api_key) - 1:
            self.current_key_index += 1
            new_api_key = self.api_key[self.current_key_index]
            logging.warning(f"Switched to next API key index={self.current_key_index}.")
            self._init_client()
            return True
        logging.error("All API keys have been exhausted.")
        return False

    async def run_llm(
        self,
        model: str,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        stream: bool = True,
    ) -> Optional[Any]:
        """
        Returns a streaming async iterator when stream=True, compatible with:
            async for chunk in response_stream:
                chunk.choices[0].delta.content
        """
        while self.current_key_index < len(self.api_key):
            for attempt in range(1, self.max_retries + 1):
                try:
                    if self.provider == "together":
                        response = await asyncio.wait_for(
                            self.client.chat.completions.create(
                                model=model,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                stream=stream,
                            ),
                            timeout=self.timeout,
                        )
                        logging.info(f"Together API call successful on attempt {attempt}.")
                        return response

                    if self.provider == "openrouter":
                        # OpenAI-compatible call
                        response = await asyncio.wait_for(
                            self.client.chat.completions.create(
                                model=model,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                stream=stream,
                            ),
                            timeout=self.timeout,
                        )
                        logging.info(f"OpenRouter API call successful on attempt {attempt}.")
                        return response

                    raise ValueError(f"Unsupported provider: {self.provider}")

                except asyncio.TimeoutError:
                    if attempt < self.max_retries:
                        sleep_time = self.retry_backoff[min(attempt - 1, len(self.retry_backoff) - 1)]
                        logging.warning(f"LLM call timed out. Retrying in {sleep_time}s (Attempt {attempt}/{self.max_retries})...")
                        await asyncio.sleep(sleep_time)
                    else:
                        logging.error(f"LLM call timed out after {self.max_retries} attempts.")

                except Exception as e:
                    # Provider-specific credit-limit strings vary; keep it robust
                    msg = str(e)
                    credit_hit = ("credit" in msg.lower() and "limit" in msg.lower()) or ("insufficient" in msg.lower())

                    if credit_hit:
                        logging.error(f"API key may be exhausted/limited: {e}")
                        if not self.switch_api_key():
                            return None
                        break  # switch key then retry outer while

                    if attempt < self.max_retries:
                        sleep_time = self.retry_backoff[min(attempt - 1, len(self.retry_backoff) - 1)]
                        logging.warning(f"LLM API error: {e}. Retrying in {sleep_time}s (Attempt {attempt}/{self.max_retries})...")
                        await asyncio.sleep(sleep_time)
                    else:
                        logging.error(f"LLM API error after {self.max_retries} attempts: {e}.")

            else:
                if not self.switch_api_key():
                    return None

        logging.error("All API keys have been exhausted.")
        return None
