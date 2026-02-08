# src/experts/llm_expert.py
from dataclasses import dataclass
from typing import Optional, List, Dict
from .base import BaseExpert, ExpertMeta
from src.provider.openrouter_client import OpenRouterClient

@dataclass
class LLMExpertConfig:
    model: str
    system_prompt: str
    temperature: float = 0.2
    max_tokens: int = 1024

class OpenRouterLLMExpert(BaseExpert):
    def __init__(
        self,
        meta: ExpertMeta,
        cfg: LLMExpertConfig,
        client: Optional[OpenRouterClient] = None,
    ):
        super().__init__(meta)
        self.cfg = cfg
        self.client = client or OpenRouterClient()

    def infer(self, prompt: str) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": prompt},
        ]
        out = self.client.chat(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return out["text"]

    def infer_with_usage(self, prompt: str) -> Dict:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": prompt},
        ]
        out = self.client.chat(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return out
