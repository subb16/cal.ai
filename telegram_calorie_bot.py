import os
import re
import sqlite3
from datetime import datetime, date
from pathlib import Path
import json
from typing import Optional, Tuple, Dict, Any, List

from rapidfuzz import process, fuzz
from telegram import Update, ForceReply
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from llm_food_normalizer import normalize_food_text
import logging
import csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# --- Global in-memory cache for today ---
live_today_cache = {}  # { user_id: { date_str: [entries] } }
USER_BASE_DIR = Path('users_data')

# --------------------
# Telegram handlers
# --------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    user_dir = USER_BASE_DIR / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)
    await update.message.reply_text(
        "Hi! I can help you track your calories and macronutrients. Send me what you ate (e.g. '2 eggs and 1 cup of rice').\nA personal folder has been set up for your logs!"
    )

CSV_PATH = Path("db.csv")

FIELDNAMES = ["date", "user_id", "calories", "protein", "carbs", "fat", "history"]

def _load_all_rows() -> List[Dict[str, Any]]:
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def _save_all_rows(rows: List[Dict[str, Any]]):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def update_user_day(date_str: str, user_id: str, new_cals: float, new_protein: float, new_carbs: float, new_fat: float, new_food: str):
    # Load, update (or create), save back
    all_rows = _load_all_rows()
    for row in all_rows:
        if row["date"] == date_str and row["user_id"] == user_id:
            row["calories"] = str(float(row["calories"]) + new_cals)
            row["protein"] = str(float(row.get("protein",0)) + new_protein)
            row["carbs"] = str(float(row.get("carbs",0)) + new_carbs)
            row["fat"] = str(float(row.get("fat",0)) + new_fat)
            if row["history"]:
                row["history"] += "; " + new_food
            else:
                row["history"] = new_food
            _save_all_rows(all_rows)
            return
    # Not found: make new
    all_rows.append({
        "date": date_str,
        "user_id": user_id,
        "calories": str(new_cals),
        "protein": str(new_protein),
        "carbs": str(new_carbs),
        "fat": str(new_fat),
        "history": new_food,
    })
    _save_all_rows(all_rows)

def get_user_day(date_str: str, user_id: str) -> Optional[Dict[str,Any]]:
    all_rows = _load_all_rows()
    for row in all_rows:
        if row["date"] == date_str and row["user_id"] == user_id:
            return row
    return None

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
    user_dir = USER_BASE_DIR / f"user_{user_id}"
    jsonl_path = user_dir / f"{dt}.jsonl"

    logger.info("Processing message from user %s: %s", user_id, text)
    try:
        logger.info("Calling LLM normalizer for user %s ...", user_id)
        llm_items = normalize_food_text(text)
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
    reply = meal_line + "\n" + day_line
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

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    print("Bot started. Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == "__main__":
    main()
