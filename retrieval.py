from typing import Optional, List, Dict
from rapidfuzz import fuzz 
import logging
import json
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import re
from functools import lru_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
KB_GLOBAL_PATH = Path("knowledge_base.jsonl")

def save_kb_entries(entries: List[Dict[str, Any]]):
    with open(KB_GLOBAL_PATH, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def add_kb_entry(text: str) -> int:
    load_kb_entries_cached.cache_clear()
    entries = load_kb_entries()
    next_id = max((entry.get("id", 0) for entry in entries), default=0) + 1
    entry = {"id": next_id, "text": text.strip()}
    entries.append(entry)
    save_kb_entries(entries)
    return next_id


def delete_kb_entry(entry_id: int) -> bool:
    load_kb_entries_cached.cache_clear()
    entries = load_kb_entries()
    new_entries = [entry for entry in entries if entry.get("id") != entry_id]
    if len(new_entries) == len(entries):
        return False
    save_kb_entries(new_entries)
    return True

def load_kb_entries() -> List[Dict[str, Any]]:
    if not KB_GLOBAL_PATH.exists():
        return []
    entries = []
    with open(KB_GLOBAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries

@lru_cache(maxsize=1)
def load_kb_entries_cached():
    return load_kb_entries()

def normalize_text(s: str) -> str:
    # Lowercase and remove extra symbols/spaces for more stable matching
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_kb_index(entries: List[Dict]) -> List[Dict]:
    """Add 'name' and 'name_norm' to each entry."""
    for entry in entries:
        text = entry.get("text", "") or ""
        # Assume name is everything before first comma
        name = text.split(",")[0].strip()
        entry["name"] = name
        entry["name_norm"] = normalize_text(name)
    return entries

def retrieve_kb_context(
    query: str,
    top_k: int = 3,
    min_score: int = 35,
) -> Optional[str]:
    entries = load_kb_entries_cached()
    if not entries:
        logger.info("KB retrieval: No KB entries found in knowledge_base.jsonl")
        return None

    # Enrich entries with name + normalized name
    entries = build_kb_index(entries)

    query_norm = normalize_text(query)
    if not query_norm:
        logger.warning("KB retrieval: Empty/invalid query %r", query)
        return None

    scores: List[tuple[int, Dict]] = []

    for entry in entries:
        name_norm = entry.get("name_norm", "")
        if not name_norm:
            continue

        # Fuzzy score on names only
        score_fuzzy = fuzz.token_set_ratio(query_norm, name_norm)

        # Simple token overlap bonus (helps with 'gnc bar' vs 'gnc wafer protein bar')
        q_tokens = set(query_norm.split())
        e_tokens = set(name_norm.split())
        overlap = len(q_tokens & e_tokens)
        bonus = min(overlap * 5, 15)  # overlap 1–3 tokens → +5 to +15

        score = int(score_fuzzy + bonus)
        scores.append((score, entry))

        logger.info(
            "KB match: query=%r vs name=%r → fuzzy=%0.1f, bonus=%d, total=%d",
            query,
            entry.get("name"),
            score_fuzzy,
            bonus,
            score,
        )

    if not scores:
        logger.warning("KB retrieval: No entries to score for query %r", query)
        return None

    # Sort: highest score first, then longer name (more specific) first
    scores.sort(
        key=lambda t: (t[0], len(t[1].get("name", ""))),
        reverse=True,
    )

    best_score = scores[0][0]
    # Dynamic cutoff: keep entries reasonably close to the best one
    dynamic_cutoff = max(min_score, int(best_score * 0.7))

    filtered = [t for t in scores if t[0] >= dynamic_cutoff]
    selected_entries = [entry for _, entry in filtered][:top_k]

    if not selected_entries:
        logger.warning(
            "KB retrieval: No entries matched (query=%r, best_score=%d, cutoff=%d)",
            query,
            best_score,
            dynamic_cutoff,
        )
        return None

    # If query is very short/generic and there are multiple very high matches, pick just the best
    words = query_norm.split()
    if len(selected_entries) > 1 and len(words) <= 2:
        top_score = scores[0][0]
        second_score = scores[1][0] if len(scores) > 1 else 0
        if top_score >= 90 and second_score == top_score:
            logger.info(
                "KB retrieval: generic short query %r with multiple perfect matches, "
                "using only best match",
                query,
            )
            selected_entries = [selected_entries[0]]

    # Build context string exactly like your original
    lines = [
        f"- Note #{entry['id']}: {entry.get('text', '')}"
        for entry in selected_entries
    ]
    context = "\n".join(lines)

    logger.info(
        "✅ KB retrieval: Found %d matching entries for query %r (best_score=%d, cutoff=%d)",
        len(selected_entries),
        query,
        best_score,
        dynamic_cutoff,
    )
    logger.info("KB context being sent to LLM:\n%s", context)

    return context
