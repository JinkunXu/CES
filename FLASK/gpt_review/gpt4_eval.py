import sys
sys.path.append("../")

import argparse
import json
import os
import logging
import re
import ast
import asyncio
from typing import Any, Dict, List, Optional

import shortuuid

# Original concurrent OpenAI evaluator
from openai_concurrent import OpenAIChatCompletionConcurrent

# OpenRouter / OpenAI compatible client (async)
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def parse_score(review, num):
    try:
        match = re.findall(r'{[^}]+}', review)
        if len(match) > 0:
            dictionary_part = match[-1].replace("\n", "").replace('_', " ").lower()
            lines = ast.literal_eval(dictionary_part)
            for key, value in lines.items():
                if value == 'na':
                    lines[key] = 'N/A'
                elif value == 'n/a':
                    lines[key] = 'N/A'
                elif value == 'not applicable':
                    lines[key] = 'N/A'
            return lines
        else:
            return {}
    except Exception as e:
        logger.error(f'{e}\nContent: {review}\n'
                     'You must manually fix the score pair.')
        return {}


def gen_prompt(reviewer_jsons, prompt_jsons, skills_jsons, response, item):
    reviewer_idx = 1
    prompt_id = reviewer_jsons[reviewer_idx]['prompt_id']
    prompt_json = prompt_jsons[prompt_id - 1]
    assert prompt_json['prompt_id'] == prompt_id

    sys_prompt = prompt_json['system_prompt']
    prompt_template = prompt_json['prompt_template']
    defaults = prompt_json['defaults']

    # skills = metrics
    skills = ""
    metric_list = item["skill"]
    for label in metric_list:
        for skill in skills_jsons:
            if label in skill["Skill"]:
                name = skill["Skill"]
                criteria = skill["Criteria"]
                skills += f"\n{name}: {criteria}"
                scoring = skill["Scoring"]
                skills += f"\nScore 1: {scoring['1']}"
                skills += f"\nScore 2: {scoring['2']}"
                skills += f"\nScore 3: {scoring['3']}"
                skills += f"\nScore 4: {scoring['4']}"
                skills += f"\nScore 5: {scoring['5']}\n\n"
                break

    prompt = prompt_template.format(
        question=item["instruction"],
        response=response,
        skills=skills,
        num=3,
        sample_answer=item["answer"],
        **defaults
    )
    return sys_prompt, prompt


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    file_extension = file_path.split('.')[-1]
    if file_extension == "jsonl":
        with open(file_path, 'r', encoding="utf-8") as f:
            json_list = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                json_list.append(json.loads(line))
            return json_list
    else:
        with open(file_path, 'r', encoding="utf-8") as f:
            return json.load(f)


async def _openrouter_create_many(
    requests: List[Dict[str, Any]],
    review_model: str,
    max_tokens_default: int,
    parallel: int,
    extra_headers: Dict[str, str],
) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    """
    Run chat.completions concurrently against OpenRouter using AsyncOpenAI.
    Returns:
      - responses: list of dicts in shape {"response": <OpenAIResponse>, "question_id": ..., "review_id": ...}
      - fails: list of dicts {"question_id": ..., "review_id": ..., "error": "..."}
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Please export OPENROUTER_API_KEY.")

    client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    sem = asyncio.Semaphore(parallel)

    async def one_call(req: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=review_model,
                    messages=req["request"]["messages"],
                    temperature=req.get("temperature", 0),
                    max_tokens=req.get("max_tokens", max_tokens_default),
                    extra_headers=extra_headers,
                )
                return {
                    "ok": True,
                    "question_id": req["question_id"],
                    "review_id": req["review_id"],
                    "response": resp,
                }
            except Exception as e:
                return {
                    "ok": False,
                    "question_id": req["question_id"],
                    "review_id": req["review_id"],
                    "error": repr(e),
                }

    tasks = [one_call(r) for r in requests]
    results = await asyncio.gather(*tasks)

    responses = []
    fails = []
    for r in results:
        if r["ok"]:
            responses.append({
                "response": r["response"],
                "question_id": r["question_id"],
                "review_id": r["review_id"],
            })
        else:
            fails.append({
                "question_id": r["question_id"],
                "review_id": r["review_id"],
                "error": r["error"],
            })
    return responses, fails


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--key-file', default='../openai_info/api_info.json')
    parser.add_argument('-q', '--question-file', default='../evaluation_set/flask_evaluation.jsonl')
    parser.add_argument('-s', '--skillset-file', default='../metadata_annotation/skillset/src/skillset_description.json')
    parser.add_argument('-a', '--answer-file', default='../model_output/outputs/chatgpt.jsonl')
    parser.add_argument('-p', '--prompt-file', default='src/prompt.jsonl')
    parser.add_argument('-r', '--reviewer-file', default='src/reviewer.jsonl')
    parser.add_argument('-o', '--output-review-file', default='outputs/chatgpt_review.jsonl')
    parser.add_argument('-e', '--output-error-file', default='outputs/chatgpt_review_error.jsonl')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')

    # NEW: provider + review model + parallel (for openrouter path)
    parser.add_argument(
        '--provider',
        type=str,
        default='openai',
        choices=['openai', 'openrouter'],
        help="Which provider to use for the reviewer calls: openai (original) or openrouter."
    )
    parser.add_argument(
        '--review-model',
        type=str,
        default='openai/gpt-4',
        help="Reviewer model name. For openrouter, use OpenRouter model ids like 'openai/gpt-4-0613' or 'openai/gpt-4o-mini'."
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=8,
        help="Max concurrent OpenRouter calls (only used when --provider openrouter)."
    )
    args = parser.parse_args()

    # Load resources
    key_jsons = get_json_list(args.key_file)
    question_jsons = get_json_list(args.question_file)
    skills_jsons = get_json_list(args.skillset_file)
    answer_jsons = get_json_list(args.answer_file)
    reviewer_jsons = get_json_list(args.reviewer_file)
    prompt_jsons = get_json_list(args.prompt_file)

    # Build requests
    review_jsons = []
    total_len = len(question_jsons)
    question_idx_list = list(range(total_len))
    question_copy = []
    answer_copy = []

    requests_payload = []
    for i in question_idx_list:
        answer_elem = None
        for row in answer_jsons:
            if row.get('question_id') == question_jsons[i]['idx']:
                answer_elem = row
                break

        if answer_elem is None:
            # No matching answer for this question
            logger.warning(f"Missing answer for question idx={question_jsons[i]['idx']}. Skipping.")
            continue

        answer_copy.append(answer_elem)
        assert answer_copy[-1]['question_id'] == question_jsons[i]['idx']
        question_copy.append(question_jsons[i])

        sys_prompt, prompt = gen_prompt(
            reviewer_jsons, prompt_jsons, skills_jsons,
            answer_copy[-1]["text"], question_jsons[i]
        )

        review_id = shortuuid.uuid()
        review_jsons.append({
            'review_id': review_id,
            'question_id': question_jsons[i]['idx'],
            'metadata': {},
        })

        # Note: keep the same structure as original script
        requests_payload.append(
            {
                'review_id': review_id,
                'question_id': question_jsons[i]['idx'],
                'metadata': {},
                'request': {
                    "model": args.review_model,
                    "messages": [
                        {'role': 'system', 'content': sys_prompt},
                        {'role': 'user', 'content': prompt},
                    ]
                },
                # setting temperature 0 for reproducibility
                "temperature": 0,
                "max_tokens": args.max_tokens
            }
        )

    # Execute reviewer calls
    if args.provider == "openai":
        # Original behavior
        openai_concurrent = OpenAIChatCompletionConcurrent(
            api_keys=key_jsons["api_keys"],
            requests_per_minute=180,
            expected_response_seconds=5
        )
        responses, fails = openai_concurrent.create_many(requests_payload)

    elif args.provider == "openrouter":
        extra_headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_X_TITLE", "flask-gpt-review"),
        }
        responses, fails = asyncio.run(
            _openrouter_create_many(
                requests=requests_payload,
                review_model=args.review_model,
                max_tokens_default=args.max_tokens,
                parallel=args.parallel,
                extra_headers=extra_headers,
            )
        )

    else:
        raise ValueError(f"Unsupported provider: {args.provider}")

    print(f"[Evaluator] provider={args.provider}, review_model={args.review_model}")
    # Extract reviews + tokens
    reviews = []
    total_tokens = []
    for r in responses:
        try:
            resp = r["response"]
            reviews.append(resp.choices[0].message.content)
            # usage may be missing in some edge cases; default to 0
            usage = getattr(resp, "usage", None)
            total_tokens.append(getattr(usage, "total_tokens", 0) if usage is not None else 0)
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            reviews.append("")
            total_tokens.append(0)

    print("total_token:", sum(total_tokens))

    # Ensure output directories exist
    output_directory = os.path.dirname(args.output_error_file)
    if output_directory and (not os.path.exists(output_directory)):
        os.makedirs(output_directory, exist_ok=True)

    review_out_dir = os.path.dirname(args.output_review_file)
    if review_out_dir and (not os.path.exists(review_out_dir)):
        os.makedirs(review_out_dir, exist_ok=True)

    # Handle fails: write error items
    delete_index = []
    if len(fails) > 0:
        with open(f'{args.output_error_file}', 'w', encoding="utf-8") as output_error_file:
            try:
                for fail in fails:
                    qid = fail.get("question_id")
                    print("fail:", fail)
                    delete_elem_idx = None
                    for index, item in enumerate(question_copy):
                        # question_copy items use 'idx'
                        if int(item.get("idx")) == int(qid):
                            delete_elem_idx = index
                            break
                    if delete_elem_idx is not None:
                        delete_index.append(delete_elem_idx)
                        output_error_file.write(json.dumps(question_copy[delete_elem_idx], ensure_ascii=False) + '\n')
            except Exception as e:
                print("@@@ error writing fails:", e)
                delete_index = []

    # Remove failed ones
    question_copy = [item for index, item in enumerate(question_copy) if index not in delete_index]

    # Write review output (jsonl)
    with open(f'{args.output_review_file}', 'a', encoding="utf-8") as output_review_file:
        for idx, review in enumerate(reviews):
            scores = parse_score(review, num=3)
            # review_jsons aligns with requests_payload order; if fails removed, we rely on sorting later
            if idx >= len(review_jsons):
                continue

            review_jsons[idx] = question_copy[idx] if idx < len(question_copy) else review_jsons[idx]

            # attach target_txt from answer_jsons
            for row in answer_jsons:
                if row.get('question_id') == review_jsons[idx].get('idx'):
                    review_jsons[idx]['target_txt'] = row["text"]
                    break

            review_jsons[idx]['review'] = review
            review_jsons[idx]['score'] = scores
            review_jsons[idx]['total_tokens_step4'] = total_tokens[idx]

            try:
                output_review_file.write(json.dumps(review_jsons[idx], ensure_ascii=False) + '\n')
            except Exception as e:
                output_review_file.write('\n')
                print("write_error_idx:", review_jsons[idx].get('idx'), e)

    # Sort output by idx
    with open(f'{args.output_review_file}', 'r', encoding="utf-8") as output_read_file:
        lines = output_read_file.readlines()
    json_objects = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            json_objects.append(json.loads(line))
        except Exception:
            continue

    sorted_objects = sorted(json_objects, key=lambda obj: obj.get('idx'))

    with open(f'{args.output_review_file}', 'w', encoding="utf-8") as output_write_file:
        for obj in sorted_objects:
            try:
                output_write_file.write(json.dumps(obj, ensure_ascii=False) + '\n')
            except Exception as e:
                output_write_file.write('\n')
                print("sort_write_error_idx:", obj.get('idx'), e)
