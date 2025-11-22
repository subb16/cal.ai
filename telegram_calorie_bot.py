import os
from datetime import datetime, date
from pathlib import Path
import json
import re
from typing import Optional, Tuple, Dict, Any, List

from rapidfuzz import process, fuzz
from telegram import Update, ForceReply
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from llm_food_normalizer import normalize_food_text
from retrieval import load_kb_entries, retrieve_kb_context, save_kb_entries, add_kb_entry, delete_kb_entry
import logging
import csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# Suppress noisy HTTP request logs from httpx/telegram
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# --- Global in-memory cache for today ---
live_today_cache = {}  # { user_id: { date_str: [entries] } }
USER_BASE_DIR = Path('users_data')
TARGETS_FILE = Path('user_targets.json')


def ensure_user_dir(user_id: str) -> Path:
    path = USER_BASE_DIR / f"user_{user_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_user_targets() -> Dict[str, float]:
    """Load user target calories from JSON file."""
    if not TARGETS_FILE.exists():
        return {}
    try:
        with open(TARGETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_user_targets(targets: Dict[str, float]):
    """Save user target calories to JSON file."""
    with open(TARGETS_FILE, "w", encoding="utf-8") as f:
        json.dump(targets, f, indent=2)


def get_user_target(user_id: str) -> Optional[float]:
    """Get target calories for a user. Returns None if not set."""
    targets = load_user_targets()
    return targets.get(user_id)


def set_user_target(user_id: str, target_cal: float):
    """Set target calories for a user."""
    targets = load_user_targets()
    targets[user_id] = target_cal
    save_user_targets(targets)



# --------------------
# Telegram handlers
# --------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    ensure_user_dir(user_id)
    await update.message.reply_text(
        "Hi! I can help you track your calories and macronutrients. Send me what you ate (e.g. '2 eggs and 1 cup of rice').\nA personal folder has been set up for your logs!"
    )


def safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default

def aggregate_today(user_id: str, dt: str = None):
    if dt is None:
        dt = date.today().isoformat()
    user_dir = USER_BASE_DIR / f"user_{user_id}"
    jsonl_path = user_dir / f"{dt}.jsonl"
    entries = []
    total_cal = total_prot = total_carbs = total_fat = 0.0
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    entries.append((idx, obj))
                    total_cal += float(obj.get("total_kcal") or obj.get("kcal") or 0)
                    total_prot += float(obj.get("protein") or 0)
                    total_carbs += float(obj.get("carbs") or 0)
                    total_fat += float(obj.get("fat") or 0)
                except Exception as e:
                    continue
    return total_cal, total_prot, total_carbs, total_fat, entries

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    user_id = str(update.effective_user.id)
    dt = date.today().isoformat()
    user_dir = ensure_user_dir(user_id)
    jsonl_path = user_dir / f"{dt}.jsonl"
    kb_context = build_kb_context_for_message(text)
    
    if kb_context:
        logger.info("📚 KB context retrieved and will be sent to LLM")
        logger.info("KB context:\n%s", kb_context)
    else:
        logger.info("⚠️ No KB context found for this query")

    logger.info("Processing message from user %s: %s", user_id, text)
    try:
        logger.info("Calling LLM normalizer for user %s ...", user_id)
        llm_items = normalize_food_text(text, kb_context=kb_context)
        logger.info("LLM returned %d items", len(llm_items) if llm_items else 0)
        logger.debug("Full LLM raw output: %s", llm_items)
    except Exception as e:
        logger.exception("❌ LLM call failed: %s", e)
        await update.message.reply_text(
            "❌ Error calling the LLM (HF API).\nCheck your HF_TOKEN or your internet.\n\n"
            f"Error details:\n{e}"
        )
        return

    if not llm_items or all(
        (not obj.get("item") or obj.get("item","") in ["", "hi", "hello", "hey"])
        for obj in llm_items
    ):
        await update.message.reply_text(
            "⚠️ I didn't detect any food in your message. Please describe your meal!"
        )
        return

    # Append entries to today's JSONL log
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for obj in llm_items:
            if not obj.get("item") or obj.get("item","") in ["", "hi", "hello", "hey"]:
                continue
            live_today_cache.setdefault(user_id, {}).setdefault(dt, []).append(obj)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Aggregate today's totals for reply
    total_cal, total_prot, total_carbs, total_fat, entries = aggregate_today(user_id, dt)
    # Aggregate just this meal
    meal_cal = meal_prot = meal_carbs = meal_fat = 0.0
    for obj in llm_items:
        meal_cal += float(obj.get("total_kcal") or obj.get("kcal") or 0)
        meal_prot += float(obj.get("protein") or 0)
        meal_carbs += float(obj.get("carbs") or 0)
        meal_fat += float(obj.get("fat") or 0)
    # Make a short reply: log and day total
    meal_line = f"🍽️ *This meal*: {meal_cal:.0f} kcal, Protein: {meal_prot:.1f} g, Carbs: {meal_carbs:.1f} g, Fat: {meal_fat:.1f} g"
    day_line = f"📅 *Day total*: {total_cal:.0f} kcal, Protein: {total_prot:.1f} g, Carbs: {total_carbs:.1f} g, Fat: {total_fat:.1f} g"
    
    # Add remaining calories if target is set
    target_cal = get_user_target(user_id)
    remaining_lines = []
    if target_cal is not None:
        remaining = target_cal - total_cal
        if remaining >= 0:
            remaining_lines.append(f"🎯 *Remaining*: {remaining:.0f} kcal ({total_cal:.0f}/{target_cal:.0f})")
        else:
            remaining_lines.append(f"⚠️ *Over by*: {abs(remaining):.0f} kcal ({total_cal:.0f}/{target_cal:.0f})")
    
    reply = meal_line + "\n" + day_line
    if remaining_lines:
        reply += "\n" + "\n".join(remaining_lines)
    await update.message.reply_text(reply, parse_mode="Markdown")

async def summary_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    dt = date.today().isoformat()
    total_cal, total_prot, total_carbs, total_fat, entries = aggregate_today(user_id, dt)
    items_list = []
    for idx, obj in entries:
        item = obj.get("item") or "?"
        qty = obj.get("quantity", "?")
        unit = obj.get("unit", "?")
        cal = obj.get("total_kcal") or obj.get("kcal") or 0
        macro = []
        for key in ("protein", "carbs", "fat"):
            v = obj.get(key)
            if v is not None:
                macro.append(f"{key}: {v}")
        items_list.append(f"{idx}. {item}, {qty} {unit} ({cal} kcal" + (", " + ", ".join(macro) if macro else "") + ")")
    summary = f"""📅 *Today's Totals:*
Calories: {total_cal:.0f} kcal
Protein: {total_prot:.1f} g
Carbs: {total_carbs:.1f} g
Fat: {total_fat:.1f} g

🍽️ *Meals today:*
""" + ("\n".join(items_list) if items_list else "None yet!")
    await update.message.reply_text(summary, parse_mode="Markdown")

def delete_user_entry_by_id(user_id: str, dt: str, entry_id: int) -> bool:
    """Remove entry by its index (1-based line) from JSONL. Returns True on success."""
    user_dir = USER_BASE_DIR / f"user_{user_id}"
    jsonl_path = user_dir / f"{dt}.jsonl"
    if not jsonl_path.exists():
        return False
    lines = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if entry_id < 1 or entry_id > len(lines):
        return False
    del lines[entry_id - 1]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return True

def clear_user_day(user_id: str, dt: str) -> bool:
    user_dir = USER_BASE_DIR / f"user_{user_id}"
    jsonl_path = user_dir / f"{dt}.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
        return True
    return False

async def delete_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    dt = date.today().isoformat()
    args = context.args
    if not args or not args[0].isdigit():
        await update.message.reply_text("Usage: /delete <entry number>. Use /summary to see the numbers.")
        return
    entry_id = int(args[0])
    ok = delete_user_entry_by_id(user_id, dt, entry_id)
    if ok:
        await update.message.reply_text(f"✅ Deleted entry #{entry_id} for today.")
    else:
        await update.message.reply_text(f"❌ Cannot delete entry #{entry_id}. Does it exist?")

async def clearall_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    dt = date.today().isoformat()
    ok = clear_user_day(user_id, dt)
    if ok:
        await update.message.reply_text(f"🗑️ Cleared ALL entries for today.")
    else:
        await update.message.reply_text(f"No entries found for today to clear.")


async def addnote_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.partition(" ")[2].strip()
    if not text:
        await update.message.reply_text("Usage: /addnote your note text")
        return
    note_id = add_kb_entry(text)
    await update.message.reply_text(f"✅ Added note #{note_id}. It will be used for everyone.")


async def notes_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    entries = load_kb_entries()
    if not entries:
        await update.message.reply_text("No notes saved yet. Use /addnote to add one.")
        return
    lines = [f"{entry['id']}. {entry['text']}" for entry in entries]
    await update.message.reply_text("📚 *Knowledge base:*\n" + "\n".join(lines), parse_mode="Markdown")


async def delnote_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args or not args[0].isdigit():
        await update.message.reply_text("Usage: /delnote <note number>. See /notes to list them.")
        return
    note_id = int(args[0])
    ok = delete_kb_entry(note_id)
    if ok:
        await update.message.reply_text(f"Deleted note #{note_id}.")
    else:
        await update.message.reply_text(f"Couldn't find note #{note_id}.")


async def settarget_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set target calories for the user."""
    user_id = str(update.effective_user.id)
    args = context.args
    if not args or not args[0].replace('.', '').isdigit():
        await update.message.reply_text("Usage: /settarget <calories>\nExample: /settarget 2000")
        return
    try:
        target_cal = float(args[0])
        if target_cal <= 0:
            await update.message.reply_text("Target calories must be greater than 0.")
            return
        set_user_target(user_id, target_cal)
        await update.message.reply_text(f"✅ Target calories set to {target_cal:.0f} kcal per day.")
    except ValueError:
        await update.message.reply_text("Invalid number. Usage: /settarget <calories>")



async def users_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not USER_BASE_DIR.exists():
        count = 0
    else:
        count = sum(1 for path in USER_BASE_DIR.iterdir() if path.is_dir())
    await update.message.reply_text(f"👥 Total users with data: {count}")


def build_kb_context_for_message(text: str) -> Optional[str]:
    """
    Split a multi-item message into chunks (by 'and' / ','), run KB retrieval
    for each chunk separately, and merge the results into a single context.
    """
    parts = re.split(r"\band\b|,", text, flags=re.IGNORECASE)
    kb_lines: list[str] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        ctx = retrieve_kb_context(part)
        if not ctx:
            continue

        # ctx may contain multiple "- Note #..." lines
        for line in ctx.splitlines():
            line = line.strip()
            if line and line not in kb_lines:
                kb_lines.append(line)

    if not kb_lines:
        return None

    return "\n".join(kb_lines)

# --------------------
# Main
# --------------------
def main():
    token = os.environ.get('TELEGRAM_TOKEN')
    if not token:
        print("Please set TELEGRAM_TOKEN environment variable.")
        return
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("summary", summary_cmd))
    app.add_handler(CommandHandler("delete", delete_cmd))
    app.add_handler(CommandHandler("clear", clearall_cmd))
    app.add_handler(CommandHandler("addtokb", addnote_cmd))
    app.add_handler(CommandHandler("showkb", notes_cmd))
    app.add_handler(CommandHandler("delkb", delnote_cmd))
    app.add_handler(CommandHandler("totalusers", users_cmd))
    app.add_handler(CommandHandler("setcal", settarget_cmd))


    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    print("Bot started. Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == "__main__":
    main()
