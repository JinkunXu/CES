# scripts/eval_mmlu_meta_openrouter.py
# This script evaluates multiple LLMs on MMLU using OpenRouter API
# and analyzes their performance across 10 capability dimensions. 
# It uses a disk cache to avoid redundant API calls, and logs errors/parse failures for diagnosis.
import os, json, argparse, hashlib, random
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import re
# repo import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.provider.openrouter_client import OpenRouterClient


# -------------------- Disk Cache --------------------
class DiskCache:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db: Dict[str, Any] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self.db[obj["key"]] = obj["val"]

    def _key(self, obj: Any) -> str:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def get(self, obj: Any) -> Optional[Any]:
        return self.db.get(self._key(obj))

    def put(self, obj: Any, val: Any) -> Any:
        k = self._key(obj)
        if k in self.db:
            return self.db[k]
        self.db[k] = val
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": k, "val": val}, ensure_ascii=False) + "\n")
        return val


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# -------------------- 10-dim capability mapping --------------------
CAP_NAMES = [
    "math",              # 0
    "physics_chem",      # 1
    "bio_medicine",      # 2
    "cs_ml_security",    # 3
    "law_politics",      # 4
    "business_econ",     # 5
    "history_humanities",# 6
    "psych_sociology",   # 7
    "logic_ethics",      # 8
    "geography_other",   # 9
]

SUBJECT2CAP = {
    # math
    "abstract_algebra": 0,
    "college_mathematics": 0,
    "elementary_mathematics": 0,
    "high_school_mathematics": 0,
    "high_school_statistics": 0,
    "econometrics": 0,

    # physics/chem
    "astronomy": 1,
    "college_physics": 1,
    "high_school_physics": 1,
    "conceptual_physics": 1,
    "college_chemistry": 1,
    "high_school_chemistry": 1,

    # bio/medicine
    "anatomy": 2,
    "college_biology": 2,
    "high_school_biology": 2,
    "clinical_knowledge": 2,
    "college_medicine": 2,
    "professional_medicine": 2,
    "medical_genetics": 2,
    "nutrition": 2,
    "virology": 2,
    "human_aging": 2,
    "human_sexuality": 2,

    # CS/ML/security/EE
    "college_computer_science": 3,
    "high_school_computer_science": 3,
    "computer_security": 3,
    "machine_learning": 3,
    "electrical_engineering": 3,

    # law/politics
    "international_law": 4,
    "jurisprudence": 4,
    "professional_law": 4,
    "high_school_government_and_politics": 4,
    "public_relations": 4,
    "us_foreign_policy": 4,
    "security_studies": 4,

    # business/econ/management
    "business_ethics": 5,
    "management": 5,
    "marketing": 5,
    "professional_accounting": 5,
    "high_school_macroeconomics": 5,
    "high_school_microeconomics": 5,

    # history/humanities
    "philosophy": 6,
    "prehistory": 6,
    "high_school_european_history": 6,
    "high_school_us_history": 6,
    "high_school_world_history": 6,
    "world_religions": 6,

    # psych/soc
    "high_school_psychology": 7,
    "professional_psychology": 7,
    "sociology": 7,

    # logic/ethics
    "formal_logic": 8,
    "logical_fallacies": 8,
    "moral_disputes": 8,
    "moral_scenarios": 8,

    # geography/other
    "high_school_geography": 9,
    "global_facts": 9,
    "miscellaneous": 9,
}

def subject_to_cap(subject: str) -> int:
    return SUBJECT2CAP.get(subject, 9)


# -------------------- MMLU prompt & parsing --------------------
CHOICE_LETTERS = ["A", "B", "C", "D"]

def format_mmlu_prompt(question: str, choices: List[str]) -> str:
    lines = [question.strip(), ""]
    for i, c in enumerate(choices):
        lines.append(f"{CHOICE_LETTERS[i]}. {c}")
    lines.append("")
    lines.append("Answer with a single letter: A, B, C, or D.")
    return "\n".join(lines)




_PATTERNS = [
    # 1) 纯字母 / 前缀形式
    re.compile(r"^\s*([ABCD])\s*$", re.I),
    re.compile(r"^\s*([ABCD])[\.\)]\s*", re.I),
    re.compile(r"^\s*[\(\[]\s*([ABCD])\s*[\)\]]\s*$", re.I),

    # 2) 常见标签：Answer / Final / Choice
    re.compile(r"\bANSWER\s*[:\-]?\s*[\(\[]?\s*([ABCD])\s*[\)\]]?\b", re.I),
    re.compile(r"\bFINAL\s*[:\-]?\s*[\(\[]?\s*([ABCD])\s*[\)\]]?\b", re.I),
    re.compile(r"\bCHOICE\s*[:\-]?\s*[\(\[]?\s*([ABCD])\s*[\)\]]?\b", re.I),

    # 3) 英文句式
    re.compile(r"\bTHE\s+ANSWER\s+IS\s+[\(\[]?\s*([ABCD])\s*[\)\]]?\b", re.I),
]

def parse_choice_letter(text: str) -> str | None:
    if not text:
        return None
    t = text.strip()

    # 如果是明显的错误信息，直接认为无效（避免污染统计）
    if t.startswith("[") and "ERROR" in t.upper():
        return None

    for pat in _PATTERNS:
        m = pat.search(t)
        if m:
            return m.group(1).upper()

    # 兜底：找“独立字母”而不是字符串包含
    m = re.search(r"(?<![A-Z0-9])([ABCD])(?![A-Z0-9])", t.upper())
    return m.group(1) if m else None



def get_gold_letter(ds, ex) -> str:
    """
    cais/mmlu 的 answer 是 ClassLabel；datasets 通常返回 int(0..3)
    也可能返回 "A"/"B"/"C"/"D"。这里统一成字母。
    """
    ans = ex["answer"]
    if isinstance(ans, int):
        return CHOICE_LETTERS[int(ans)]
    if isinstance(ans, str):
        a = ans.strip().upper()
        if a in CHOICE_LETTERS:
            return a
        # 有些展示会像 "1B" 这种（极少见），兜底取最后一个字母
        for ch in reversed(a):
            if ch in CHOICE_LETTERS:
                return ch
    # 兜底
    return "A"


# -------------------- Evaluation --------------------
def eval_one_model_on_mmlu(
    client: OpenRouterClient,
    cache: DiskCache,
    model: str,
    examples: List[Dict[str, Any]],
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    err_log: str,
    log_ok_every: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    returns:
      cap_correct[10], cap_total[10]
    """
    cap_correct = np.zeros(len(CAP_NAMES), dtype=np.int64)
    cap_total = np.zeros(len(CAP_NAMES), dtype=np.int64)

    for idx, ex in enumerate(tqdm(examples, desc=f"eval {model}", leave=False)):
        q = ex["question"]
        choices = ex["choices"]
        subject = ex["subject"]
        cap = subject_to_cap(subject)

        prompt = format_mmlu_prompt(q, choices)
        gold = get_gold_letter(None, ex)

        key = {
            "type": "mmlu_mcq",
            "model": model,
            "system": system_prompt,
            "prompt": prompt,
        }

        cached = cache.get(key)
        if cached is None:
            out = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if not out.get("ok", False):
                append_jsonl(err_log, {
                    "event": "generation_error",
                    "model": model,
                    "subject": subject,
                    "cap": int(cap),
                    "idx": int(idx),
                    "status_code": out.get("status_code"),
                    "request_id": out.get("request_id"),
                    "error_type": out.get("error_type"),
                    "error_msg": out.get("error_msg"),
                    "response_body": out.get("response_body"),
                    "traceback": out.get("traceback"),
                    "prompt_preview": prompt[:400],
                })
                text = ""  # 失败不计入 cache
            else:
                text = (out.get("text") or "").strip()
                # 成功也可能是空输出：也要记录
                if text == "":
                    append_jsonl(err_log, {
                        "event": "empty_output",
                        "model": model,
                        "subject": subject,
                        "cap": int(cap),
                        "idx": int(idx),
                        "prompt_preview": prompt[:400],
                    })
                else:
                    cache.put(key, text)

                # 可选：采样记录成功输出，便于快速定位“输出格式是否异常”
                if log_ok_every and (idx % log_ok_every == 0):
                    append_jsonl(err_log, {
                        "event": "ok_sample",
                        "model": model,
                        "subject": subject,
                        "cap": int(cap),
                        "idx": int(idx),
                        "text_preview": text[:200],
                    })
        else:
            text = cached

        pred = parse_choice_letter(text)

        # parse_fail 也写日志（这类会导致 0 分，但原因是“格式/空输出/错误”）
        if pred is None and text != "":
            append_jsonl(err_log, {
                "event": "parse_fail",
                "model": model,
                "subject": subject,
                "cap": int(cap),
                "idx": int(idx),
                "gold": gold,
                "text_preview": (text or "")[:300],
            })

        cap_total[cap] += 1
        if pred is not None and pred == gold:
            cap_correct[cap] += 1


    return cap_correct, cap_total


def sample_examples(ds_all, n_per_subject: int, seed: int) -> List[Dict[str, Any]]:
    """
    从 all split 里按 subject 分层抽样，保证每个 subject 有近似数量样本。
    """
    rng = random.Random(seed)
    by_subj: Dict[str, List[Dict[str, Any]]] = {}
    for ex in ds_all:
        s = ex["subject"]
        by_subj.setdefault(s, []).append(ex)

    out: List[Dict[str, Any]] = []
    for s, xs in by_subj.items():
        rng.shuffle(xs)
        take = xs[: min(n_per_subject, len(xs))]
        out.extend(take)

    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="test", choices=["dev", "validation", "test"])
    ap.add_argument("--n_per_subject", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--cache", type=str, default="runs/cache_mmlu_eval.jsonl")
    ap.add_argument("--out", type=str, default="artifacts/mmlu_meta_10d.json")

    # 推理参数
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=16)


    ap.add_argument("--err_log", type=str, default="runs/mmlu_meta_eval_err.jsonl",
                help="Write per-call errors/parse_fails to this jsonl.")
    ap.add_argument("--log_ok_every", type=int, default=0,
                help="If >0, log every N successful calls too (for debugging).")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cache = DiskCache(args.cache)
    client = OpenRouterClient()

    # 1) load MMLU (use config="all" then stratified sampling)
    ds = load_dataset("cais/mmlu", "all")[args.split]
    examples = sample_examples(ds, n_per_subject=args.n_per_subject, seed=args.seed)

    # 2) define experts
    EXPERTS = [
        # (name, openrouter_model)
        ("qwen-3", "qwen/qwen3-vl-235b-a22b-instruct"),
        ("deepseek-r1", "deepseek/deepseek-r1"),
        ("llama-3.1", "meta-llama/llama-3.1-405b-instruct"),
        ("deepseek-3", "deepseek/deepseek-v3.2"),
        ("claude-4.5", "anthropic/claude-sonnet-4.5"),
        ("gpt-4o", "openai/gpt-4o"),
        ("Mistral-8x22B", "mistralai/mixtral-8x22b-instruct"),
        ("mistral-small-creative", "mistralai/mistral-small-creative"),
        ("grok-4.1-fast", "x-ai/grok-4.1-fast"),
        ("qwen-2.5-72b", "qwen/qwen-2.5-72b-instruct"),
        ("gemma-2-27b-it", "google/gemma-2-27b-it"),
        ("qwen-2.5-32b-instruct", "qwen/qwen-2.5-32b-instruct"),
        ("glm-4-32b", "z-ai/glm-4-32b"),
        ("llama-3.1-70b-instruct", "meta-llama/llama-3.1-70b-instruct"),
        ("gpt-3.5-turbo", "openai/gpt-3.5-turbo"),
    ]

    sys_prompt = (
        "You are taking a multiple-choice test. "
        "You MUST answer with a single letter: A, B, C, or D. "
        "No explanation, no extra text."
    )

    results = {
        "cap_names": CAP_NAMES,
        "split": args.split,
        "n_per_subject": args.n_per_subject,
        "seed": args.seed,
        "models": {},
    }

    # 3) eval
    for name, model in tqdm(EXPERTS, desc="models"):
        cap_correct, cap_total = eval_one_model_on_mmlu(
        client=client,
        cache=cache,
        model=model,
        examples=examples,
        system_prompt=sys_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        err_log=args.err_log,
        log_ok_every=args.log_ok_every,
    )


        if name in ["gpt-5", "gemini-3"]:
            out = client.chat(
                model=model,
                messages=[{"role":"system","content":sys_prompt},
                        {"role":"user","content":"1+1=?\nA.1\nB.2\nC.3\nD.4\nAnswer with a single letter."}],
                temperature=0.0,
                max_tokens=16,
            )
            print(name, out.get("ok"), out.get("error"), repr((out.get("text") or "")[:200]))

        cap_acc = (cap_correct / np.maximum(cap_total, 1)).tolist()
        overall = float(cap_correct.sum() / max(int(cap_total.sum()), 1))

        results["models"][name] = {
            "openrouter_model": model,
            "cap_acc_10d": cap_acc,     
            "overall_acc": overall,
            "cap_correct": cap_correct.tolist(),
            "cap_total": cap_total.tolist(),
        }

        
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[DONE] wrote meta to: {args.out}")

        


if __name__ == "__main__":
    main()
