import os
import json
import argparse
import pandas as pd
from typing import List, Dict, Any, Tuple, Union
from alpaca_eval import evaluate

os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_KEY"] = "YOUR OPENROUTER_API_KEY"  # set your OpenRouter API key in environment variable
def _normalize_model_row(d: Dict[str, Any], generator_name: str, idx: int) -> Dict[str, Any]:
    inst = d.get("instruction") or d.get("prompt") or d.get("query")
    out = d.get("output") or d.get("response") or d.get("completion")

    
    inst = "" if inst is None else str(inst)
    out = "" if out is None else str(out)

    
    inst = inst.strip()
    out = out.strip()

    if not inst:
        inst = " "
    if not out:
        out = " "

    return {
        "instruction": inst,
        "output": out,
        "generator": d.get("generator", generator_name),
    }



def load_model_outputs(path: str, generator_name: str) -> pd.DataFrame:
    """
    Supports:
      - .jsonl: one JSON object per line
      - .json: a JSON list of objects, or a single JSON object
    Also tolerates blank lines, BOM, and prints useful diagnostics.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    content_stripped = content.strip()
    if not content_stripped:
        raise ValueError(f"Empty model_outputs file: {path}")

    rows: List[Dict[str, Any]] = []

    # Case A: JSON list / dict (most .json files)
    if content_stripped[0] in "[{":
        try:
            obj = json.loads(content_stripped)
            if isinstance(obj, list):
                for i, d in enumerate(obj):
                    if not isinstance(d, dict):
                        raise ValueError(f"JSON list element {i} is not a dict: {type(d)}")
                    rows.append(_normalize_model_row(d, generator_name, i))
            elif isinstance(obj, dict):
                rows.append(_normalize_model_row(obj, generator_name, 0))
            else:
                raise ValueError(f"Unsupported JSON top-level type: {type(obj)}")
            return pd.DataFrame(rows)
        except json.JSONDecodeError:
            # Fall through to jsonl parsing attempt
            pass

    # Case B: JSONL (fallback): parse line by line with good error message
    lines = content.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        try:
            d = json.loads(s)
        except json.JSONDecodeError as e:
            # show the offending line prefix for debugging
            prefix = s[:200]
            raise ValueError(
                f"JSONL decode failed at line {i+1}: {e}\n"
                f"Offending line (first 200 chars): {prefix}"
            )
        if not isinstance(d, dict):
            raise ValueError(f"JSONL line {i+1} is not a dict: {type(d)}")
        rows.append(_normalize_model_row(d, generator_name, len(rows)))

    if not rows:
        raise ValueError(f"No valid JSON objects found in: {path}")

    return pd.DataFrame(rows)


def load_reference_json(path: str) -> pd.DataFrame:
    df = pd.read_json(path, orient="records")
    required = {"instruction", "output"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Reference file missing columns: {missing}. got {list(df.columns)}")

    if "generator" not in df.columns:
        df["generator"] = "reference_model"

    return df[["instruction", "output", "generator"]]


def align_by_instruction(model_df: pd.DataFrame, ref_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # de-duplicate
    if model_df["instruction"].duplicated().any():
        dup = int(model_df["instruction"].duplicated().sum())
        print(f"[warn] model_outputs has {dup} duplicated instructions; keeping first.")
        model_df = model_df.drop_duplicates(subset=["instruction"], keep="first")

    if ref_df["instruction"].duplicated().any():
        dup = int(ref_df["instruction"].duplicated().sum())
        print(f"[warn] reference_outputs has {dup} duplicated instructions; keeping first.")
        ref_df = ref_df.drop_duplicates(subset=["instruction"], keep="first")

    merged = model_df.merge(ref_df, on="instruction", how="inner", suffixes=("", "_ref"))

    n_model, n_ref, n_aligned = len(model_df), len(ref_df), len(merged)
    print(f"[align] model rows: {n_model}")
    print(f"[align] ref rows:   {n_ref}")
    print(f"[align] aligned:    {n_aligned}")
    print(f"[align] dropped(no ref):   {n_model - n_aligned}")
    print(f"[align] dropped(no model): {n_ref - n_aligned}")

    aligned_model = merged[["instruction", "output", "generator"]].copy()
    aligned_ref = merged[["instruction", "output_ref", "generator_ref"]].copy()
    aligned_ref.columns = ["instruction", "output", "generator"]

    return aligned_model, aligned_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_outputs", required=True, help="jsonl or json (list) file with outputs")
    ap.add_argument("--reference_outputs", required=True, help="json list file: {instruction, output, generator}")
    ap.add_argument("--out_dir", default="leaderboard", help="output dir for alpaca_eval artifacts")
    ap.add_argument("--annotators_config", default="alpaca_eval", help="alpaca_eval annotator config name")
    ap.add_argument("--generator_name", default="your_model", help="fallback generator name for model outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model_df = load_model_outputs(args.model_outputs, args.generator_name)
    ref_df = load_reference_json(args.reference_outputs)

    aligned_model, aligned_ref = align_by_instruction(model_df, ref_df)

    result = evaluate(
        model_outputs=aligned_model,
        reference_outputs=aligned_ref,
        annotators_config=args.annotators_config,
        output_path=args.out_dir,
    )

    print("\n==== AlpacaEval Result ====")
    print(result)


if __name__ == "__main__":
    main()
