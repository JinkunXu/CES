# src/provider/openrouter_client.py
import os
import time
import traceback
from typing import List, Dict, Any, Optional

class OpenRouterClient:
    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("Please install: pip install openai")

        self.api_key = os.environ.get("OPENROUTER_API_KEY")

       
        timeout_s = float(os.environ.get("OPENROUTER_TIMEOUT", "60"))

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            timeout=timeout_s,
        )

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        retries: int = 3,
        backoff: float = 1.8,
    ) -> Dict[str, Any]:
        """
        Returns a structured dict:
          ok: bool
          text: str
          usage: {prompt_tokens, completion_tokens, total_tokens}
          status_code: Optional[int]
          request_id: Optional[str]
          error_type / error_msg / traceback (if ok=False)
        """
        if not self.api_key:
            return {
                "ok": False,
                "text": "",
                "usage": {},
                "status_code": None,
                "request_id": None,
                "error_type": "MissingAPIKey",
                "error_msg": "Missing OPENROUTER_API_KEY",
                "traceback": None,
            }

        headers = extra_headers or {
            "HTTP-Referer": "https://github.com/EntroShape/Online_LLM_Selection",
            "X-Title": "LLM-Bandit-Router",
        }

        last_err: Optional[Dict[str, Any]] = None

        for k in range(max(1, retries)):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=headers,
                    extra_body=extra_body,
                )


                text = completion.choices[0].message.content if completion.choices else ""


                usage_obj = completion.usage


                        
                usage_dict = {}
                if usage_obj is not None:
                    if hasattr(usage_obj, "model_dump"):
                        usage_dict = usage_obj.model_dump()
                    else:
                        usage_dict = dict(getattr(usage_obj, "__dict__", {}) or {})

                usage = {
                    "prompt_tokens": int(usage_dict.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(usage_dict.get("completion_tokens", 0) or 0),
                    "total_tokens": int(usage_dict.get("total_tokens", 0) or 0),
                }


                        
                cost = float(usage_dict.get("cost", 0.0) or 0.0)
                usage["cost"] = cost


                return {
                    "ok": True,
                    "text": text or "",
                    "usage": usage, 
                    "id": getattr(completion, "id", None), 
                    "status_code": None,
                    "request_id": None,
                }

            except Exception as e:
                tb = traceback.format_exc()
                err_type = type(e).__name__
                status_code = getattr(e, "status_code", None)

                response_body = None
                request_id = None
                try:
                    resp = getattr(e, "response", None)
                    if resp is not None:
                        status_code = getattr(resp, "status_code", status_code)
                        # headers / request id
                        headers_obj = getattr(resp, "headers", None)
                        if headers_obj:
                            request_id = headers_obj.get("x-request-id") or headers_obj.get("X-Request-Id")
                        # body
                        response_body = getattr(resp, "text", None) or getattr(resp, "content", None)
                except Exception:
                    pass

                last_err = {
                    "ok": False,
                    "text": "",
                    "usage": {},
                    "status_code": status_code,
                    "request_id": request_id,
                    "error_type": err_type,
                    "error_msg": str(e),
                    "response_body": response_body,
                    "traceback": tb,
                }

                
                sleep_s = (backoff ** k) + 0.2 * (k + 1)
                time.sleep(min(sleep_s, 20.0))

        return last_err or {
            "ok": False,
            "text": "",
            "usage": {},
            "status_code": None,
            "request_id": None,
            "error_type": "UnknownError",
            "error_msg": "unknown",
            "traceback": None,
        }
