# src/evaluators/llm_judge.py
import json, re
from dataclasses import dataclass
from typing import Optional, Dict, List
from src.provider.openrouter_client import OpenRouterClient

@dataclass
class JudgeConfig:
    model: str
    system_prompt: str
    temperature: float = 0.0
    max_tokens: int = 512

def _extract_json(text: str) -> Optional[Dict]:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

class LLMJudge:
    def __init__(self, cfg: JudgeConfig, client: Optional[OpenRouterClient] = None):
        self.cfg = cfg
        self.client = client or OpenRouterClient()

    def score(self, prompt: str, answer: str, reference: Optional[str] = None) -> float:
        if reference is None:
            user = (
                "Rate answer quality from 0 to 1.\n"
                "Return STRICT JSON: {\"score\": float, \"reason\": str}\n\n"
                f"PROMPT:\n{prompt}\n\nANSWER:\n{answer}\n"
            )
        else:
            user = (
                "Rate how well ANSWER matches REFERENCE (correctness+completeness) from 0 to 1.\n"
                "Return STRICT JSON: {\"score\": float, \"reason\": str}\n\n"
                f"PROMPT:\n{prompt}\n\nANSWER:\n{answer}\n\nREFERENCE:\n{reference}\n"
            )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": user},
        ]
        out = self.client.chat(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )

        obj = _extract_json(out["text"])
        if not obj or "score" not in obj:
            return 0.0
        try:
            s = float(obj["score"])
            return float(max(0.0, min(1.0, s)))
        except Exception:
            return 0.0
