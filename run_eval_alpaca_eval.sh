cd CES

#python scripts/eval_mmlu_meta.py  # evaluate on MMLU if u add new models
python scripts/alpaca_generate_reference.py # generate reference responses for evaluation

python scripts/run_alpaca_eval.py --k 3 

python extract_alpacaeval_outputs.py --router_log "YOUR_ROUTER_OUTPUT.jsonl" --out "FINAL_OUTPUT.json" --genertator_name "CES" # extract the final responses from the router log, and save to FINAL_OUTPUT.json

python eval_alpacaeval.py --model_outputs "FINAL_OUTPUT.json" --reference_outputs "REFERENCE_RESPONSES.json" # evaluate the final responses against the reference responses, and print the results