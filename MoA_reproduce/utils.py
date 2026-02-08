import os
import json
import time
import requests
import openai
import copy

from loguru import logger

# OpenRouter additions
from typing import List, Dict, Optional
from openai import OpenAI as OpenAIClient

DEBUG = int(os.environ.get("DEBUG", "0"))

# OpenRouter constants
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def generate_together(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    streaming=False,
):

    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        try:

            endpoint = "https://api.together.xyz/v1/chat/completions"

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}`."
                )

            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}",
                },
            )
            if "error" in res.json():
                logger.error(res.json())
                if res.json()["error"]["type"] == "invalid_request_error":
                    logger.info("Input + output is longer than max_position_id.")
                    return None

            output = res.json()["choices"][0]["message"]["content"]

            break

        except Exception as e:
            logger.error(e)
            if DEBUG:
                logger.debug(f"Msgs: `{messages}`")

            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:
        return output

    output = output.strip()

    if DEBUG:
        logger.debug(f"Output: `{output[:20]}...`.")

    return output


def generate_together_stream(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    endpoint = "https://api.together.xyz/v1"
    client = openai.OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"), base_url=endpoint
    )
    endpoint = "https://api.together.xyz/v1/chat/completions"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature if temperature > 1e-4 else 0,
        max_tokens=max_tokens,
        stream=True,  # this time, we set stream=True
    )

    return response


def generate_openai(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output


def generate_openrouter(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    extra_headers: Optional[Dict[str, str]] = None,
    **kwargs,
) -> str:
    """
    OpenRouter chat.completions wrapper.
    Reads:
      - OPENROUTER_API_KEY (required)
      - OPENROUTER_HTTP_REFERER (optional, default http://localhost)
      - OPENROUTER_X_TITLE (optional, default moa-flask)
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Please export OPENROUTER_API_KEY."
        )

    client = OpenAIClient(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    if extra_headers is None:
        extra_headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_X_TITLE", "moa-flask"),
        }

    output = None
    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}` via OpenRouter."
                )

            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=(temperature if temperature > 1e-4 else 0),
                max_tokens=max_tokens,
                extra_headers=extra_headers,
            )

            output = (resp.choices[0].message.content or "").strip()
            break

        except Exception as e:
            logger.error(e)
            if DEBUG:
                logger.debug(f"Msgs: `{messages}`")
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    return output


def inject_references_to_messages(
    messages,
    references,
):

    messages = copy.deepcopy(messages)

    system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

    for i, reference in enumerate(references):
        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + system
    else:
        messages = [{"role": "system", "content": system}] + messages

    return messages


def generate_with_references(
    model,
    messages,
    references=[],
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_together,
):

    if len(references) > 0:
        messages = inject_references_to_messages(messages, references)

    return generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
