import os
import json
import logging
from typing import List, Dict, Any

from huggingface_hub import InferenceClient


logger = logging.getLogger(__name__)
_HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
_client: InferenceClient | None = None



def _get_client() -> InferenceClient:
    """Create/reuse a single HF inference client using featherless-ai."""
    global _client
    if _client is None:
        token = os.environ.get("HF_TOKEN") or "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"
        if not token:
            raise RuntimeError("HF_TOKEN environment variable not set")
        _client = InferenceClient(
            api_key=token,
            provider="featherless-ai",
        )
        logger.info("Initialized HF InferenceClient with provider featherless-ai")
    return _client


def _build_messages(user_text: str):
    """
    Build chat messages for Mistral. We force JSON-only output.
    """
    system_prompt = """
You are a strict assistant that helps find out calories and macronutrients of the food items in a user's message.

Given a user's message describing what they ate, you MUST output ONLY a single JSON ARRAY of food items with their calories and macronutrients (no other text).

The JSON array should look like this:
[
  {"item": "egg", "quantity": 2, "unit": "piece","total_kcal": 155, "protein": 13.0, "carbs": 1.1, "fat": 11.0},
  {"item": "cooked rice", "quantity": 1, "unit": "cup", "total_kcal": 130, "protein": 2.7, "carbs": 28.0, "fat": 0.3},
]

Rules:
1. Output MUST be valid JSON and nothing else.
2. If multiple foods, output multiple objects.
3. If unsure about quantity/grams, make a reasonable guess.
4. Use lowercase for item names.
5. Prefer 'piece' for generic items like fruit; 'cup'/'bowl' for volume.
""".strip()

    user_prompt = f"Generate macronutrients and calories breakdown for the following food items:\n{user_text}\n\n Return ONLY a JSON array as described."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_text_from_chat_response(resp) -> str:
    """
    Extract the assistant text from ChatCompletionOutput.
    """
    try:
        text = resp.choices[0].message["content"]
        return text if isinstance(text, str) else str(text)
    except Exception as e:
        logger.exception("Failed to extract text from chat_completion response: %s", e)
        return ""


def normalize_food_text(user_text: str) -> List[Dict[str, Any]]:
    """
    Use Mistral chat_completion to convert free text into a list of objects:

    Returns [] if no parseable JSON is found.
    """
    client = _get_client()
    messages = _build_messages(user_text)

    logger.info("Calling Mistral chat_completion for food normalization...")
    resp = client.chat_completion(
        model=_HF_MODEL_ID,
        messages=messages,
        max_tokens=256,
        temperature=0.1,
    )

    raw_text = _extract_text_from_chat_response(resp).strip()
    logger.info("Raw LLM output (first 1000 chars): %s", raw_text[:1000])

    # Try direct JSON parse
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            logger.info("Successfully parsed JSON array directly from LLM.")
            return data
    except Exception:
        logger.debug("Direct json.loads failed; trying substring extraction.")


    logger.warning("LLM did not return a parseable JSON array. Raw output: %s", raw_text[:2000])
    return []
