# src/aggregators/llm_aggregator.py
from dataclasses import dataclass
from typing import List, Dict, Optional
from src.provider.openrouter_client import OpenRouterClient

@dataclass
class AggregatorConfig:
    model: str
    system_prompt: str
    temperature: float = 0.0
    max_tokens: int = 1024

class LLMAggregator:
    def __init__(self, cfg: AggregatorConfig, client: Optional[OpenRouterClient] = None):
        self.cfg = cfg
        self.client = client or OpenRouterClient()

    def aggregate(self, user_prompt: str, expert_names: List[str], expert_answers: List[str]) -> str:
        packed = []
        for n, a in zip(expert_names, expert_answers):
            packed.append(f"## Expert: {n}\n{a}\n")

        user = (
            "User question:\n"
            f"{user_prompt}\n\n"
            "Expert proposals:\n" + "\n".join(packed) +
            "\n\nYou are an expert integrator. Produce the best final answer to the user.\n"
            "Your role is to:"
            "- Understand the user's intent and main question(s) by carefully reviewing their query."
            "- Evaluate expert inputs, preserving their quality opinions while ensuring relevance, accuracy, and alignment with the user's needs."
            "-Resolve any contradictions or gaps logically, combining expert insights into a single, unified response."                
            "- Synthesize the most appropriate information into a clear, actionable, and user-friendly answer."
            "Add your own insight if needed to enhance the final output."
            "Your response must prioritize clarity, accuracy, and usefulness, ensuring it directly addresses the user's needs while"
            "retaining the value of expert contributions."
            "Avoid referencing the integration process or individual experts."
            "Do NOT mention other experts or the integration process.\n"
            "Keep the response concise and correct."
            "Task:\n"
            "1) Read the user question.\n"
            "2) Review the proposals (may contain errors).\n"
            "3) Provide the best final answer."
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
        return out["text"]
