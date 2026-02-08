# src/data/dataset_loader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterator, Tuple, List
from datasets import load_dataset

def _build_prompt_from_fields(ex: Dict[str, Any], fields: List[str], sep: str = "\n\n") -> str:
    parts = []
    for f in fields:
        v = ex.get(f, None)
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        parts.append(str(v))
    return sep.join(parts).strip()

def iter_dataset(
    dataset: str,
    limit: Optional[int] = None,
    seed: int = 0,
    # generic HF config (for non-builtins)
    dataset_hf_path: Optional[str] = None,
    dataset_config: Optional[str] = None,
    dataset_split: Optional[str] = None,
    dataset_text_fields: Optional[str] = None,  # comma-separated
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Yield (prompt, raw_example) for different datasets.

    Supported presets:
      - alpacaeval2
      - mtbench
      - flask_hard (generic HF loader)
      - generic (requires dataset_hf_path + fields)

    Notes:
      - MT-Bench comes in multiple formats across repos.
        This function supports a common HF packaging. If it fails, use generic mode.

    """
    ds_name = dataset.lower().strip()

    # ---------------- AlpacaEval 2.0 ----------------
    if ds_name in ["alpacaeval2", "alpaca_eval2", "alpaca_eval", "alpacaeval"]:
        ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        for ex in ds:
            instruction = ex["instruction"]
            inp = ex.get("input") or ""
            prompt = instruction if inp.strip() == "" else f"{instruction}\n\nInput:\n{inp}"
            yield prompt, dict(ex)
        return

    # ---------------- MT-Bench (FastChat-style HF dump) ----------------
    # There are multiple variants; this one is a common public HF format.
    if ds_name in ["mtbench", "mt-bench", "mt_bench"]:
        # Try a common dataset that contains the questions.
        # If your environment doesn't have it, fall back to generic mode.
        try:
            # Many MT-Bench dumps contain a "question" list with multi-turn.
            # We'll take turn1 only unless you want multi-turn prompting.
            ds = load_dataset("lmsys/mt_bench_human_judgments")["question"]
        except Exception as e:
            raise RuntimeError(
                "Cannot load MT-Bench preset via 'lmsys/mt_bench_human_judgments'.\n"
                "Use generic mode instead:\n"
                "  --dataset generic --dataset_hf_path <path> --dataset_split <split> "
                "--dataset_text_fields <field1,field2,...>\n"
                f"Original error: {repr(e)}"
            )

        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        for ex in ds:
            # typical schema: {"question_id":..., "category":..., "turns":[...]}
            turns = ex.get("turns") or ex.get("turn") or ex.get("prompt") or None
            if isinstance(turns, list) and len(turns) > 0:
                # Use first turn by default; you can extend to multi-turn later.
                prompt = str(turns[0]).strip()
            elif isinstance(turns, str):
                prompt = turns.strip()
            else:
                # last resort: try common fields
                prompt = _build_prompt_from_fields(ex, ["instruction", "question", "input", "text"])
            if not prompt:
                continue
            yield prompt, dict(ex)
        return

    # ---------------- FLASK-Hard (use generic HF config) ----------------
    if ds_name in ["flask-hard", "flask_hard", "flaskhard"]:
        # FLASK-Hard has multiple HF wrappers; use generic args to avoid repo-specific code.
        # You must pass dataset_hf_path / split / text_fields (or rely on defaults below).
        hf_path = dataset_hf_path or "tatsu-lab/flask"  # placeholder default; adjust if you have a specific repo
        split = dataset_split or "test"
        cfg = dataset_config  # may be None

        ds = load_dataset(hf_path, cfg)[split] if cfg is not None else load_dataset(hf_path)[split]
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        # default: try common fields
        fields = (
            [s.strip() for s in dataset_text_fields.split(",")]
            if dataset_text_fields
            else ["instruction", "input", "question", "prompt"]
        )

        for ex in ds:
            prompt = _build_prompt_from_fields(ex, fields)
            if not prompt:
                continue
            yield prompt, dict(ex)
        return

    # ---------------- Generic HF dataset ----------------
    if ds_name in ["generic", "hf"]:
        if not dataset_hf_path:
            raise ValueError("--dataset generic requires --dataset_hf_path")
        split = dataset_split or "test"
        cfg = dataset_config
        ds = load_dataset(dataset_hf_path, cfg)[split] if cfg is not None else load_dataset(dataset_hf_path)[split]
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        if not dataset_text_fields:
            raise ValueError("--dataset generic requires --dataset_text_fields (comma-separated fields)")
        fields = [s.strip() for s in dataset_text_fields.split(",") if s.strip()]

        for ex in ds:
            prompt = _build_prompt_from_fields(ex, fields)
            if not prompt:
                continue
            yield prompt, dict(ex)
        return

    raise ValueError(f"Unknown dataset='{dataset}'. Supported: alpacaeval2 | mtbench | flask_hard | generic")
