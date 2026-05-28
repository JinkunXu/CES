# [KDD'2026] CES: Combinatorial Experts Selection via Contextual Linear Bandits

CES is a cost-aware expert selection framework for routing each prompt to a small subset of LLM experts, aggregating their answers, and updating the router online from evaluation feedback. This repository contains scripts to reproduce the CES experiments on AlpacaEval 2.0, MT-Bench, FLASK-Hard, and objective multiple-choice benchmarks.

## Repository Layout

```text
CES/
  src/                         Core router, bandit, expert, aggregator, embedding code
  scripts/                     CES experiment and evaluation entry points
  artifacts/mmlu_meta_10d.json Offline model capability profiles used by CES
  FastChat/                    MT-Bench judge data and FastChat evaluation utilities
  FLASK/                       FLASK evaluation utilities
  MoA_reproduce/               Optional Mixture-of-Agents baseline scripts
  KABB_reproduce/              Optional KABB baseline scripts
  data/alpaca_eval/            AlpacaEval reference outputs
  runs/, outputs/              Generated logs, caches, and evaluation outputs
```

## Environment

The scripts were developed with Python 3.12. A clean conda environment is recommended.

```bash
conda create -n ces python=3.12 -y
conda activate ces
cd CES

python -m pip install -U pip
python -m pip install \
  numpy pandas tqdm datasets sentence-transformers torch openai alpaca-eval \
  fschat shortuuid fire loguru pyyaml matplotlib scikit-learn scipy tiktoken together
```

If you want to use the bundled FastChat copy instead of the PyPI package, install it in editable mode:

```bash
python -m pip install -e FastChat
```

The first run downloads the default sentence-transformer encoder:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

If your machine is offline during experiments, pre-download/cache this model first.

## API Keys

All CES generation and judging calls use OpenRouter through an OpenAI-compatible client. Set credentials before running experiments:

```bash
export OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"
export OPENROUTER_TIMEOUT=120

# Needed by AlpacaEval/FastChat-style judge code.
export OPENAI_API_BASE="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
```

Optional metadata headers:

```bash
export OPENROUTER_HTTP_REFERER="https://github.com/<your-name>/CES"
export OPENROUTER_X_TITLE="CES"
```

Never commit real API keys. Before publishing the repository, remove generated caches/logs that may contain raw model outputs or usage metadata if you do not want to release them.

## Quick Smoke Test

Run a tiny AlpacaEval 2.0 job to check the environment, OpenRouter key, embedding model, cache writing, expert calls, aggregation, and online reward update:

```bash
python scripts/run_alpacaeval2.py \
  --limit 2 \
  --seed 0 \
  --k 2 \
  --lam_cost 0.01 \
  --out_log runs/smoke/alpacaeval2_ces.jsonl \
  --cache runs/smoke/cache_alpacaeval2_ces.jsonl
```

Expected output files:

```text
runs/smoke/alpacaeval2_ces.jsonl
runs/smoke/cache_alpacaeval2_ces.jsonl
```

## Reproduce CES Experiments

### 1. AlpacaEval 2.0

Generate CES responses:

```bash
python scripts/run_alpacaeval2.py \
  --limit 805 \
  --seed 0 \
  --k 2 \
  --lam_cost 0.01 \
  --out_log runs/alpacaeval2/CES.jsonl \
  --cache runs/alpacaeval2/cache_CES.jsonl
```

Convert the router log to AlpacaEval format:

```bash
python extract_alpacaeval_outputs.py \
  --router_log runs/alpacaeval2/CES.jsonl \
  --out runs/alpacaeval2/CES_output.jsonl \
  --generator_name CES
```

Evaluate with the provided reference outputs:

```bash
python eval_alpaca.py \
  --model_outputs runs/alpacaeval2/CES_output.jsonl \
  --reference_outputs data/alpaca_eval/reference.json \
  --out_dir runs/alpacaeval2_eval/CES \
  --generator_name CES
```

If you need to regenerate reference outputs, edit `MODEL_NAME` and `OUTPUT_PATH` in `scripts/alpaca_generate_reference.py`, then run:

```bash
python scripts/alpaca_generate_reference.py
```

### 2. MT-Bench

Generate MT-Bench answers in FastChat format:

```bash
python scripts/run_mt_bench.py \
  --limit 80 \
  --seed 0 \
  --k 2 \
  --lam_cost 0.01 \
  --model_id CES \
  --out_log runs/mt_bench/CES.jsonl \
  --cache runs/mt_bench/cache_CES.jsonl \
  --mtbench_answer_dir runs/mt_bench/model_answer
```

Judge the answers:

```bash
python eval_mt_bench.py \
  --model_list CES \
  --judge-model gpt-4 \
  --mode single \
  --parallel 8
```

Show scores:

```bash
python show_mt_bench_result.py \
  --model-list CES \
  --judge-model gpt-4 \
  --mode single
```

Outputs are written under:

```text
runs/mt_bench/model_answer/CES.jsonl
outputs/mt_bench/model_judgment/gpt-4_single.jsonl
```

### 3. FLASK-Hard

Generate CES outputs:

```bash
python scripts/run_flask.py \
  --dataset flask_hard \
  --limit 200 \
  --seed 0 \
  --k 2 \
  --lam_cost 0.01 \
  --out_log outputs/flask/CES.jsonl \
  --cache outputs/flask/cache_CES.jsonl
```

Run FLASK GPT review:

```bash
cd FLASK/gpt_review
python gpt4_eval.py \
  --provider openrouter \
  --review-model openai/gpt-4o \
  --parallel 8 \
  -q ../evaluation_set/flask_hard_evaluation.jsonl \
  -a ../../outputs/flask/CES.jsonl \
  -o ../../outputs/flask/chatgpt_review.jsonl

python aggregate_skill.py -m ../../outputs/flask/chatgpt_review.jsonl
cd ../..
```

The aggregated FLASK skill table is written by the FLASK scripts under their `outputs/stats/` path.

### 4. Objective MCQ Benchmarks

Run CES on ARC-Challenge:

```bash
python scripts/run_objective_mcq.py \
  --dataset arc_challenge \
  --split validation \
  --limit 200 \
  --seed 0 \
  --k 2 \
  --lam_cost 0.01 \
  --out_log runs/objective_mcq/ces_arc_challenge.jsonl \
  --cache runs/objective_mcq/cache_ces_arc_challenge.jsonl
```

The script prints final `accuracy`, `parse_rate`, `avg_reward`, `total_cost`, and `cost_per_question`. It also stores per-example records in the JSONL log.

Supported datasets:

```text
mmlu
arc_challenge
arc_easy
commonsenseqa
hellaswag
arc_agi2 / arc-agi2
```

To evaluate each single expert on ARC-Challenge for comparison:

```bash
python scripts/eval_arc_challenge_models.py \
  --split validation \
  --limit 200 \
  --models all \
  --out-log runs/objective_mcq/arc_challenge_models.jsonl \
  --summary-out runs/objective_mcq/arc_challenge_model_summary.json \
  --summary-csv runs/objective_mcq/arc_challenge_model_summary.csv \
  --cache runs/objective_mcq/cache_arc_challenge_models.jsonl
```

## Ablation Suite

Use `scripts/run_ces_ablation_suite.py` to generate or execute consistent ablation commands. Without `--execute`, it only prints commands.

```bash
python scripts/run_ces_ablation_suite.py \
  --dataset alpacaeval2 \
  --limit 805 \
  --seed 0 \
  --base-k 2 \
  --k-values 1 2 3 4 \
  --lam-cost 0.01 \
  --out-dir runs/alpacaeval2_ablation_outputs \
  --cache-dir runs/alpacaeval2_ablation_cache
```

Run the suite:

```bash
python scripts/run_ces_ablation_suite.py \
  --dataset alpacaeval2 \
  --limit 805 \
  --seed 0 \
  --base-k 2 \
  --k-values 1 2 3 4 \
  --lam-cost 0.01 \
  --out-dir runs/alpacaeval2_ablation_outputs \
  --cache-dir runs/alpacaeval2_ablation_cache \
  --execute
```

Available variants:

```text
full
no_offline_warm_start
no_meta_vectors
no_query_embedding
no_hadamard
no_cost_penalty
k_1, k_2, k_3, k_4, ...
```

Set `--dataset mtbench` or `--dataset flask_hard` to run the same ablation grid for those benchmarks.

## Offline Capability Profiles

CES reads expert capability vectors from:

```text
artifacts/mmlu_meta_10d.json
```

If you change the expert model list, regenerate the file:

```bash
python scripts/eval_mmlu_meta.py \
  --split test \
  --n_per_subject 20 \
  --seed 0 \
  --out artifacts/mmlu_meta_10d.json \
  --cache runs/cache_mmlu_eval.jsonl
```

The router expects every expert name in the run scripts to exist in `artifacts/mmlu_meta_10d.json`.

## Optional Baselines

Baseline scripts are included under `MoA_reproduce/` and `KABB_reproduce/`. They are not required to run CES, but they can be used to reproduce comparison systems.

Example MoA command for objective MCQ:

```bash
python MoA_reproduce/generate_for_objective_mcq.py \
  --dataset arc_challenge \
  --split validation \
  --limit 200 \
  --provider openrouter \
  --model qwen/qwen-2.5-72b-instruct \
  --reference-models "qwen/qwen-2.5-32b-instruct,meta-llama/llama-3.1-70b-instruct,deepseek/deepseek-v3.2" \
  --rounds 1 \
  --out-log runs/objective_mcq/moa_arc_challenge.jsonl
```

Example KABB command for objective MCQ:

```bash
python KABB_reproduce/scripts/run_kabb_objective.py \
  --config KABB_reproduce/configs/config_template.yaml \
  --dataset arc_challenge \
  --split validation \
  --limit 200 \
  --seed 0 \
  --out-log KABB_reproduce/runs/objective_mcq/kabb_arc_challenge.jsonl
```

KABB loads `sentence-transformers/all-MiniLM-L6-v2` during import, so make sure that model is cached or that the machine has HuggingFace access.

## Caches, Costs, and Reproducibility Notes

- Each runner has a `--cache` JSONL file. Reusing it avoids repeated API calls and makes reruns cheaper.
- `--seed` controls dataset shuffling and random ablations.
- `--k` controls how many experts are selected per prompt.
- `--lam_cost` controls the cost penalty in the reward.
- `--no_query_embedding`, `--no_meta_vectors`, `--no_hadamard`, and `--no_cost_penalty` reproduce feature ablations.
- API model availability and provider-side model versions can change over time, so exact numbers may drift unless the provider pins the same model snapshots.

## Common Issues

Missing `OPENROUTER_API_KEY`:

```text
error_type: MissingAPIKey
error_msg: Missing OPENROUTER_API_KEY
```

Fix by exporting `OPENROUTER_API_KEY` before running.

MT-Bench model not found during judging:

```text
check_data(...)
```

Make sure `--model_id` in `scripts/run_mt_bench.py` matches `--model_list` in `eval_mt_bench.py`, and that the answer file exists under `runs/mt_bench/model_answer/`.

HuggingFace download failure:

```text
Failed to establish a new connection
```

Pre-download the embedding model and datasets, or set your HuggingFace mirror/proxy in the environment.
