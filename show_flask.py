import json
import math
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import matplotlib.pyplot as plt


AXIS_ORDER = [
    "robustness",
    "correctness",
    "efficiency",
    "factuality",
    "commonsense",
    "comprehension",
    "insightfulness",
    "completeness",
    "metacognition",
    "readability",
    "conciseness",
    "harmlessness",
]

# ===========
# Canonical mapping
# ===========
CANONICAL_MAP = {
    # robustness family
    "logical robustness": "robustness",
    "robustness": "robustness",

    # correctness family
    "logical correctness": "correctness",
    "correctness": "correctness",

    # efficiency family
    "logical efficiency": "efficiency",
    "efficiency": "efficiency",

    # commonsense family
    "commonsense understanding": "commonsense",
    "commonsense": "commonsense",

    # others (mostly already aligned)
    "factuality": "factuality",
    "comprehension": "comprehension",
    "insightfulness": "insightfulness",
    "completeness": "completeness",
    "metacognition": "metacognition",
    "readability": "readability",
    "conciseness": "conciseness",
    "harmlessness": "harmlessness",
}


def _norm_skill_name(s: str) -> str:
   
    return " ".join(str(s).strip().lower().split())


def _canon_skill_name(s: str) -> str:
    
    k = _norm_skill_name(s)
    return _norm_skill_name(CANONICAL_MAP.get(k, k))


def load_review_jsonl_acc(path: str) -> Dict[str, List[float]]:
    
    acc = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            score = obj.get("score", {})
            if not isinstance(score, dict):
                continue
            for k, v in score.items():
                if isinstance(v, (int, float)):
                    acc[_canon_skill_name(k)].append(float(v))
    return acc


def merge_acc(acc_list: Iterable[Dict[str, List[float]]]) -> Dict[str, List[float]]:
   
    out = defaultdict(list)
    for acc in acc_list:
        for k, vs in acc.items():
            out[k].extend(vs)
    return out


def acc_to_mean(acc: Dict[str, List[float]]) -> Dict[str, float]:
    
    mean = {}
    for k, vs in acc.items():
        if vs:
            mean[k] = sum(vs) / len(vs)
    return mean


def align_scores(mean_scores: Dict[str, float], axis_order: List[str], fill_value=float("nan")) -> List[float]:
   
    out = []
    for ax in axis_order:
        ax2 = _canon_skill_name(ax)
        out.append(mean_scores.get(ax2, fill_value))
    return out


def scan_skill_stats(paths: List[str]) -> Tuple[List[str], Dict[str, int]]:
    counts = Counter()
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                score = obj.get("score", {})
                if not isinstance(score, dict):
                    continue
                for k, v in score.items():
                    if isinstance(v, (int, float)):
                        counts[_canon_skill_name(k)] += 1
    skills_sorted = [k for k, _ in counts.most_common()]
    return skills_sorted, dict(counts)


def radar_plot(
    series: List[Tuple[str, List[float]]],
    axis_labels: List[str],
    rmin: float = 3.5,
    rmax: float = 5.0,
    title: str = "FLASK Skill Radar",
    out_path: str = "flask_radar.png",
):
    n = len(axis_labels)
    angles = [2 * math.pi * i / n for i in range(n)]
    angles += angles[:1]  # close loop

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels)
    ax.set_ylim(rmin, rmax)

    for label, vals in series:
        vals2 = vals + vals[:1]
        ax.plot(angles, vals2, linewidth=2, label=label)

    ax.set_title(title, y=1.08)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved radar figure to: {out_path}")


def _assert_files_exist(name: str, files: List[str]) -> List[str]:
    ok = []
    for f in files:
        if f and Path(f).exists():
            ok.append(f)
        else:
            print(f"[WARN] {name}: missing file: {f}")
    return ok


def main():
    model_to_review_files = {
        "GPT-4o": [
            # "FLASK/gpt_review/eval_output/GPT4o.jsonl",
        ],
        "MoA": [
            # "FLASK/gpt_review/eval_output/moa.jsonl",
        ],
        "CMoE": [
            # "FLASK/gpt_review/eval_output/cmoe.jsonl",
        ],
        "Qwen-72b": [
            # "FLASK/gpt_review/eval_output/qwen_72b.jsonl",
        ],
        "KABB": [
            # "FLASK/gpt_review/eval_output/KABB.jsonl",
        ],
    }

    USE_FIXED_AXIS_ORDER = True
    axis_labels = AXIS_ORDER if USE_FIXED_AXIS_ORDER else None  # skills_sorted 下面再决定

    
    RMIN, RMAX = 3.0, 5.0
    FILL_MISSING = RMIN  #

    
    all_existing_files = []
    cleaned_model_to_files = {}
    for name, files in model_to_review_files.items():
        ok_files = _assert_files_exist(name, files)
        cleaned_model_to_files[name] = ok_files
        all_existing_files.extend(ok_files)

    if not all_existing_files:
        raise FileNotFoundError("No review.jsonl files found. Please fill the correct paths.")

   
    skills_sorted, counts = scan_skill_stats(all_existing_files)
    print("== Skill coverage (top 50 by frequency, canonicalized) ==")
    for k in skills_sorted[:50]:
        print(f"{k:20s} {counts[k]}")

    if not USE_FIXED_AXIS_ORDER:
        axis_labels = skills_sorted  

  
    model_order = ["GPT-4o", "MoA", "CMoE", "Qwen-72b", "KABB"]
    series = []

    for name in model_order:
        files = cleaned_model_to_files.get(name, [])
        if not files:
            print(f"[WARN] skip {name}: no existing files")
            continue

        accs = [load_review_jsonl_acc(f) for f in files]
        merged = merge_acc(accs)
        mean_scores = acc_to_mean(merged)

        aligned = align_scores(mean_scores, axis_labels, fill_value=FILL_MISSING)
        series.append((name, aligned))

    if not series:
        raise RuntimeError("No valid series to plot (all models missing files).")


    radar_plot(
        series=series,
        axis_labels=axis_labels,
        rmin=RMIN,
        rmax=RMAX,
        title="Results on FLASK (available files merged) - Average Skill Scores",
        out_path="flask_radar.png",
    )

    print(f"[OK] plotted {len(series)} lines: {[name for name, _ in series]}")

if __name__ == "__main__":
    main()  