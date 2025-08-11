import os
import csv
import json
import time
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import requests

from src.data_collector import DataCollector
from src.parser import Settings

SETTINGS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "settings.json"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
ALL_VACANCIES_CSV = os.path.join(DATA_DIR, "designer_vacancies.csv")
SENT_VACANCIES_CSV = os.path.join(DATA_DIR, "sent_vacancies.csv")
FILTERED_VACANCIES_CSV = os.path.join(DATA_DIR, "designer_filtered.csv")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_dataframe(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def save_dataframe(df: pd.DataFrame, path: str, overwrite: bool = False):
    if overwrite or not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False, mode="a", header=False)


def is_schedule_ok(schedule: str) -> bool:
    if not isinstance(schedule, str):
        return False
    s = schedule.lower()
    return ("удал" in s) or ("remote" in s) or ("гибк" in s) or ("flex" in s)


def is_salary_ok(row: Dict[str, Any]) -> bool:
    # Require explicit lower bound From >= 100k RUB
    from_value = row.get("From")
    if pd.isna(from_value) or from_value is None:
        return False
    return float(from_value) >= 100_000


def is_design_role(name: str, description: str, keys: List[str]) -> bool:
    text = f"{name} {description} {' '.join(keys if isinstance(keys, list) else [])}".lower()
    design_terms = [
        "дизайн", "дизайнер", "графический", "graphic", "designer", "illustrator", "ui", "ux",
        "visual", "art director", "арт-директор", "motion", "бренд", "brand", "identity", "инфографика",
        "illustration", "figma", "photoshop", "after effects", "creative", "креатив"
    ]
    return any(term in text for term in design_terms)


def filter_vacancies(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Convert Keys from stringified list (if read from CSV) back to list for filtering
    def parse_keys(val):
        if isinstance(val, list):
            return val
        try:
            # Try JSON list
            return json.loads(val)
        except Exception:
            # Fallback: split by comma
            return [x.strip() for x in str(val).split(',') if x.strip()]

    df = df.copy()
    if "Keys" in df.columns:
        df["Keys"] = df["Keys"].apply(parse_keys)

    mask_schedule = df["Schedule"].apply(is_schedule_ok)
    mask_salary = df.apply(lambda r: is_salary_ok(r.to_dict()), axis=1)
    mask_design = df.apply(lambda r: is_design_role(str(r.get("Name", "")), str(r.get("Description", "")), r.get("Keys", [])), axis=1)
    filtered = df[mask_schedule & mask_salary & mask_design].copy()

    # Normalize published date if available
    if "PublishedAt" in filtered.columns:
        def to_dt(x):
            try:
                return datetime.fromisoformat(str(x).replace("Z", "+00:00"))
            except Exception:
                return pd.NaT
        filtered["PublishedAt_dt"] = filtered["PublishedAt"].apply(to_dt)
        filtered.sort_values(by=["PublishedAt_dt"], ascending=False, inplace=True)
    return filtered


def send_to_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; skipping send.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": False})
    if resp.status_code != 200:
        print(f"[ERROR] Telegram send failed: {resp.status_code} {resp.text}")
        return False
    return True


def pick_one_to_send(filtered_df: pd.DataFrame, sent_df: pd.DataFrame) -> Dict[str, Any]:
    # Prioritize newest not yet sent; if none, allow re-sending the newest in archive
    sent_ids = set(sent_df["Ids"]) if (not sent_df.empty and "Ids" in sent_df.columns) else set()
    for _, row in filtered_df.iterrows():
        if row["Ids"] not in sent_ids:
            return row.to_dict()
    # If all already sent, return the newest anyway
    return filtered_df.iloc[0].to_dict() if not filtered_df.empty else {}


def main():
    ensure_data_dir()

    # Prepare settings for Designer roles
    settings = Settings(SETTINGS_PATH, no_parse=True)
    # Override options to target Designer roles daily
    settings.options.update({
        "text": "Designer",
        "area": 1,
        "per_page": 50,
        "professional_roles": [4, 6, 8, 9, 34],
    })
    settings.refresh = True  # force refresh for daily run

    collector = DataCollector(settings.rates)
    vacancies = collector.collect_vacancies(query=settings.options, refresh=settings.refresh, num_workers=settings.num_workers)

    df = pd.DataFrame.from_dict(vacancies)

    # Persist all raw vacancies (append, de-dup by Ids later)
    prev_df = load_dataframe(ALL_VACANCIES_CSV)
    combined = pd.concat([prev_df, df], ignore_index=True) if not prev_df.empty else df
    combined.drop_duplicates(subset=["Ids"], keep="last", inplace=True)
    save_dataframe(combined, ALL_VACANCIES_CSV, overwrite=True)

    # Filtering logic
    filtered = filter_vacancies(combined)

    # Save filtered vacancies
    prev_filtered = load_dataframe(FILTERED_VACANCIES_CSV)
    filtered_to_save = pd.concat([prev_filtered, filtered], ignore_index=True) if not prev_filtered.empty else filtered
    filtered_to_save.drop_duplicates(subset=["Ids"], keep="last", inplace=True)
    save_dataframe(filtered_to_save, FILTERED_VACANCIES_CSV, overwrite=True)

    # Select one to send
    sent_df = load_dataframe(SENT_VACANCIES_CSV)
    chosen = pick_one_to_send(filtered, sent_df)

    if not chosen:
        print("[INFO] No suitable vacancies today.")
        return

    # Format message
    lines = [
        f"Вакансия: {chosen.get('Name', '')}",
        f"Компания: {chosen.get('Employer', '')}",
        f"График: {chosen.get('Schedule', '')}",
        f"Опыт: {chosen.get('Experience', '')}",
        (
            f"Зарплата: от {int(chosen['From']):,} руб." if not pd.isna(chosen.get('From')) else "Зарплата: не указана"
        ),
        f"Ссылка: {chosen.get('Url', '')}",
    ]
    text = "\n".join(lines)

    ok = send_to_telegram(text)

    if ok:
        sent_row = pd.DataFrame([{col: chosen.get(col) for col in combined.columns if col in chosen}])
        sent_row["SentAt"] = datetime.utcnow().isoformat()
        save_dataframe(sent_row, SENT_VACANCIES_CSV)

    print("[INFO] Daily designer vacancy processed.")


if __name__ == "__main__":
    main()