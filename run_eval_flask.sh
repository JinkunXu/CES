mkdir -p outputs/flask

# python scripts/eval_mmlu_meta.py  # evaluate on MMLU if u add new models.

python scripts/run_flask.py --k 3 --out_log "outputs/flask/YOUR_OUTPUTS.jsonl" --cache "outputs/flask/cache_CES.jsonl" 
# run the evaluation, and save the router log to outputs/flask/YOUR_OUTPUTS.jsonl

cd FLASK/gpt_review

python gpt4_eval.py \
    -a '../../outputs/flask/YOUR_OUTPUTS.jsonl' \
    -o '../../outputs/flask/chatgpt_review.jsonl'

python aggregate_skill.py -m '../../outputs/flask/chatgpt_review.jsonl'

cat outputs/stats/chatgpt_review_skill.csv