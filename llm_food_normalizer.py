import os
import json
import logging
from typing import List, Dict, Any, Optional

from huggingface_hub import InferenceClient


logger = logging.getLogger(__name__)
_HF_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
_client: InferenceClient | None = None


def _get_client() -> InferenceClient:
    """Create/reuse a single HF inference client."""
    global _client
    if _client is None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN environment variable not set")
        _client = InferenceClient(
            api_key=token,
            provider="featherless-ai"
        )
        logger.info("Initialized HF InferenceClient for model prompts")
    return _client


def _build_messages(user_text: str, kb_context: Optional[str] = None):
    system_prompt = """
    You are a strict assistant that helps calculate calories and macronutrients of the food items in a user's message.
    ...

    When KNOWLEDGE BASE entries are provided, you MUST follow these rules:

    1. If an item in the user message matches a KB item, you MUST use the KB values as the **per-unit** macros.
    2. You MUST multiply those per-unit values by the user-specified quantity.
    - Example:
        KB: "boiled egg, quantity 1, 80kcal, 6g protein, 5g fat, 0.5g carbs"
        User: "2 eggs"
        → You MUST output: total_kcal = 160, protein = 12, fat = 10, carbs = 1.0
    3. You are FORBIDDEN from using your own nutrition knowledge for items that exist in the KB.
    Use ONLY the KB values and simple multiplication.
    4. Only if an item is NOT in the KB, you may estimate its macros.

    The output must always be ONLY a JSON array of macros ( WARNING : no explanation, no markdown ) ....

    Example json output : [{
    "item": "egg",
    "quantity": 2,
    "unit": "piece",
    "total_kcal": 160,
    "protein": 12.0,
    "carbs": 1.0,
    "fat": 10.0
  }]

    """.strip()


    if kb_context:
        system_prompt += (
            "\n\nIMPORTANT - Knowledge Base Entries (USE THESE EXACT VALUES):\n"
            f"{kb_context}\n"
            "CRITICAL RULES FOR KNOWLEDGE BASE:\n"
            "1. When multiple KB entries match the user's query, pick the MOST SPECIFIC match (the one with the most matching words/details).\n"
            "2. You MUST use the exact calories and macronutrients from the knowledge base - do not estimate or guess.\n"
            "3. Scale the values by quantity if the user specifies a quantity, handle this very carefully (e.g., '2 bars' = 2x the per-bar values).\n"
        )

    user_prompt = (
        "Generate macronutrients and calories breakdown for the following food items:\n"
        f"{user_text}\n\n"
        "Return ONLY a JSON array as described and adhere to all the rules mentioned."
    )

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


def normalize_food_text(user_text: str, kb_context: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    kb_context: optional string from user knowledge base to inject into the prompt.
    Returns [] if no parseable JSON is found.
    """
    client = _get_client()
    messages = _build_messages(user_text, kb_context)
    

    logger.info("Calling Mistral chat_completion for food normalization...")
    resp = client.chat_completion(
        model=_HF_MODEL_ID,
        messages=messages,
        max_tokens=256,
        temperature=0.1,
    )

    raw_text = _extract_text_from_chat_response(resp).strip()
    logger.info("Raw LLM output (first 1000 chars): %s", raw_text)

    # Try direct JSON parse
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            logger.info("Successfully parsed JSON array directly from LLM.")
            return data
    except Exception:
        logger.debug("Direct json.loads failed; trying substring extraction.")

    # Try to extract JSON array from text (handles cases where LLM adds commentary)
    try:
        start_idx = raw_text.find('[')
        end_idx = raw_text.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_snippet = raw_text[start_idx:end_idx + 1]
            data = json.loads(json_snippet)
            if isinstance(data, list):
                logger.info("Successfully extracted and parsed JSON array from LLM output.")
                return data
    except Exception as e:
        logger.debug("JSON extraction failed: %s", e)

    logger.warning("LLM did not return a parseable JSON array. Raw output: %s", raw_text[:2000])
    return []
