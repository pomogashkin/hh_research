import argparse
import csv
import json
import os
import random
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import requests
from src.data_collector import DataCollector
from src.parser import Settings

SETTINGS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "settings.json"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
ALL_VACANCIES_CSV = os.path.join(DATA_DIR, "designer_vacancies.csv")
SENT_VACANCIES_CSV = os.path.join(DATA_DIR, "sent_vacancies.csv")
FILTERED_VACANCIES_CSV = os.path.join(DATA_DIR, "designer_filtered.csv")

TELEGRAM_BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN", "7707356102:AAFewlcvJepLrnJwdvNBZCt2wFKI44LfyF0"
)
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "8136658895")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Precompiled UI/UX exclusion pattern
_UIUX_RE = re.compile(
    r"(ui\s*/\s*ux|ux\s*/\s*ui|\bui\b|\bux\b|ui-?designer|ux-?designer|product\s+designer|ux\s*researcher|ux\s*writer)",
    re.IGNORECASE,
)

# ---------------- Utility helpers ----------------


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


def escape_html(text: str) -> str:
    if text is None:
        return ""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------- Filtering ----------------


def is_schedule_ok(schedule: str) -> bool:
    if not isinstance(schedule, str):
        return False
    s = schedule.lower()
    return ("удал" in s) or ("remote" in s) or ("гибк" in s) or ("flex" in s)


def is_salary_ok(row: Dict[str, Any], min_salary_rub: int) -> bool:
    from_value = row.get("From")
    if pd.isna(from_value) or from_value is None:
        return False
    try:
        return float(from_value) >= float(min_salary_rub)
    except Exception:
        return False


def contains_uiux(text: str) -> bool:
    return _UIUX_RE.search(text or "") is not None


def is_design_role(name: str, description: str, keys: List[str]) -> bool:
    text = f"{name} {description} {' '.join(keys if isinstance(keys, list) else [])}".lower()
    design_terms = [
        "дизайн",
        "дизайнер",
        "графический",
        "graphic",
        "designer",
        "illustrator",
        "ui",
        "ux",
        "visual",
        "art director",
        "арт-директор",
        "motion",
        "бренд",
        "brand",
        "identity",
        "инфографика",
        "illustration",
        "figma",
        "photoshop",
        "after effects",
        "creative",
        "креатив",
    ]
    return any(term in text for term in design_terms)


def is_uiux_role(name: str, description: str, keys: List[str]) -> bool:
    text = f"{name} {description} {' '.join(keys if isinstance(keys, list) else [])}"
    return contains_uiux(text)


def filter_vacancies(df: pd.DataFrame, min_salary_rub: int) -> pd.DataFrame:
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
            return [x.strip() for x in str(val).split(",") if x.strip()]

    df = df.copy()
    if "Keys" in df.columns:
        df["Keys"] = df["Keys"].apply(parse_keys)

    mask_schedule = df["Schedule"].apply(is_schedule_ok)
    mask_salary = df.apply(lambda r: is_salary_ok(r.to_dict(), min_salary_rub), axis=1)
    mask_design = df.apply(
        lambda r: is_design_role(
            str(r.get("Name", "")), str(r.get("Description", "")), r.get("Keys", [])
        ),
        axis=1,
    )
    mask_not_uiux = df.apply(
        lambda r: not is_uiux_role(
            str(r.get("Name", "")), str(r.get("Description", "")), r.get("Keys", [])
        ),
        axis=1,
    )

    filtered = df[mask_schedule & mask_salary & mask_design & mask_not_uiux].copy()

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


# ---------------- Telegram ----------------


def make_link_keyboard(url: str) -> Dict[str, Any]:
    return {"inline_keyboard": [[{"text": "Открыть вакансию", "url": url}]]}


def send_to_telegram(text: str, url_button: str | None = None) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; skipping send.")
        return False
    payload: Dict[str, Any] = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": False,
        "parse_mode": "HTML",
    }
    if url_button:
        payload["reply_markup"] = make_link_keyboard(url_button)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        print(f"[ERROR] Telegram send failed: {resp.status_code} {resp.text}")
        return False
    return True


def send_photo_to_telegram(photo_path: str, caption: str, url_button: str | None = None) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; skipping send.")
        return False
    if not os.path.isfile(photo_path):
        print(f"[WARN] Photo not found: {photo_path}; sending text message instead.")
        return send_to_telegram(caption, url_button)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    data: Dict[str, Any] = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
    if url_button:
        data["reply_markup"] = json.dumps(make_link_keyboard(url_button))
    with open(photo_path, "rb") as fh:
        resp = requests.post(url, data=data, files={"photo": fh})
    if resp.status_code != 200:
        print(f"[ERROR] Telegram photo send failed: {resp.status_code} {resp.text}")
        return False
    return True


def find_image_path(preferred_name: str) -> str:
    candidates = []
    if preferred_name:
        candidates.append(preferred_name)
        candidates.append(os.path.join(ROOT_DIR, preferred_name))
    candidates.extend(
        [
            os.path.join(ROOT_DIR, "img", preferred_name or "kuzya.jpg"),
            os.path.join(ROOT_DIR, "kuzya.jpg"),
        ]
    )
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return ""


# ---------------- Selection ----------------


def pick_one_to_send(
    filtered_df: pd.DataFrame, sent_df: pd.DataFrame, strategy: str = "new"
) -> Dict[str, Any]:
    # Prioritize newest not yet sent; if none, allow re-sending based on strategy
    def iter_rows():
        if strategy == "old":
            return filtered_df.iloc[::-1].iterrows()
        elif strategy == "random":
            idxs = list(range(len(filtered_df)))
            random.shuffle(idxs)
            for i in idxs:
                yield (filtered_df.index[i], filtered_df.iloc[i])
        else:
            return filtered_df.iterrows()

    sent_ids = set(sent_df["Ids"]) if (not sent_df.empty and "Ids" in sent_df.columns) else set()
    for _, row in iter_rows():
        if row["Ids"] not in sent_ids:
            return row.to_dict()
    # If all already sent, return according to strategy
    if filtered_df.empty:
        return {}
    if strategy == "old":
        return filtered_df.iloc[-1].to_dict()
    elif strategy == "random":
        return filtered_df.sample(1).iloc[0].to_dict()
    return filtered_df.iloc[0].to_dict()


# ---------------- Formatting ----------------


def build_caption_html(chosen: Dict[str, Any], max_desc_chars: int) -> str:
    name = escape_html(chosen.get("Name", ""))
    employer = escape_html(chosen.get("Employer", ""))
    schedule = escape_html(chosen.get("Schedule", ""))
    exp = escape_html(chosen.get("Experience", ""))
    salary_line = (
        f"Зарплата: от {int(chosen['From']):,} руб."
        if not pd.isna(chosen.get("From"))
        else "Зарплата: не указана"
    )
    salary_line = escape_html(salary_line)

    desc = str(chosen.get("Description", ""))
    desc = re.sub(r"\s+", " ", desc).strip()
    if len(desc) > max_desc_chars:
        # truncate at last space within limit
        cut = desc[: max(0, max_desc_chars - 1)]
        if " " in cut:
            cut = cut[: cut.rfind(" ")]
        desc = f"{cut}…"
    desc = escape_html(desc)

    parts = [
        f"<b>Вакансия:</b> {name}",
        f"<b>Компания:</b> {employer}",
        f"<b>Формат:</b> {schedule}",
        f"<b>Опыт:</b> {exp}",
        f"<b>ЗП:</b> {salary_line}",
        f"\n<b>Описание:</b> {desc}",
    ]
    return "\n".join(parts)


# ---------------- CLI ----------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily Designer Bot - test tools")
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not send to Telegram; just print"
    )
    parser.add_argument(
        "--force-resend", action="store_true", help="Ignore sent history for selection"
    )
    parser.add_argument(
        "--reset-sent", action="store_true", help="Clear sent history CSV before run"
    )
    parser.add_argument(
        "--choose",
        choices=["new", "old", "random"],
        default="new",
        help="Selection strategy when picking a vacancy",
    )
    parser.add_argument(
        "--send-count", type=int, default=1, help="How many vacancies to send in this run"
    )
    parser.add_argument("--min-salary", type=int, default=100000, help="Minimum From salary in RUB")
    parser.add_argument("--no-refresh", action="store_true", help="Do not refresh data from HH API")
    parser.add_argument("--text", type=str, default=None, help="Override search text")
    parser.add_argument("--area", type=int, default=None, help="Override area id (city/region)")
    parser.add_argument(
        "--image", type=str, default="kuzya.jpg", help="Path or filename of image to send as photo"
    )
    parser.add_argument("--no-photo", action="store_true", help="Do not send a photo, only text")
    parser.add_argument(
        "--caption-chars", type=int, default=400, help="Max characters for description in caption"
    )
    parser.add_argument(
        "--loop", action="store_true", help="Run continuously every day at specified time"
    )
    parser.add_argument("--at", type=str, default="10:00", help="Daily time in HH:MM (local time)")
    return parser.parse_args()


# ---------------- Job Runner ----------------


def run_job(args: argparse.Namespace):
    ensure_data_dir()

    # Prepare settings for Designer roles
    settings = Settings(SETTINGS_PATH, no_parse=True)
    # Override options to target Designer roles daily
    default_options = {
        "text": "Designer",
        "area": 1,
        "per_page": 50,
        "professional_roles": [4, 6, 8, 9, 34],
    }
    if args.text is not None:
        default_options["text"] = args.text
    if args.area is not None:
        default_options["area"] = args.area

    settings.options.update(default_options)
    settings.refresh = not args.no_refresh  # refresh by default; allow disabling

    collector = DataCollector(settings.rates)
    vacancies = collector.collect_vacancies(
        query=settings.options, refresh=settings.refresh, num_workers=settings.num_workers
    )

    df = pd.DataFrame.from_dict(vacancies)

    # Persist all raw vacancies (append, de-dup by Ids later)
    prev_df = load_dataframe(ALL_VACANCIES_CSV)
    combined = pd.concat([prev_df, df], ignore_index=True) if not prev_df.empty else df
    combined.drop_duplicates(subset=["Ids"], keep="last", inplace=True)
    save_dataframe(combined, ALL_VACANCIES_CSV, overwrite=True)

    # Filtering logic
    filtered = filter_vacancies(combined, min_salary_rub=args.min_salary)

    # Save filtered vacancies
    prev_filtered = load_dataframe(FILTERED_VACANCIES_CSV)
    filtered_to_save = (
        pd.concat([prev_filtered, filtered], ignore_index=True)
        if not prev_filtered.empty
        else filtered
    )
    filtered_to_save.drop_duplicates(subset=["Ids"], keep="last", inplace=True)
    save_dataframe(filtered_to_save, FILTERED_VACANCIES_CSV, overwrite=True)

    # Select and send
    sent_df = load_dataframe(SENT_VACANCIES_CSV)
    if args.force_resend:
        sent_df = pd.DataFrame(columns=sent_df.columns)  # treat as empty

    # Resolve image path (once per run)
    image_path = "" if args.no_photo else find_image_path(args.image)
    if image_path:
        print(f"[INFO] Using image: {image_path}")
    elif not args.no_photo:
        print(f"[WARN] Image not found for '{args.image}'. Will send text only.")

    sent_this_run: List[Dict[str, Any]] = []

    for _ in range(max(1, int(args.send_count))):
        chosen = pick_one_to_send(filtered, sent_df, strategy=args.choose)
        if not chosen:
            if _ == 0:
                print("[INFO] No suitable vacancies found for sending.")
            break

        caption = build_caption_html(chosen, max_desc_chars=args.caption_chars)
        url_btn = chosen.get("Url", "")

        print("\n--- Candidate ---\n" + re.sub(r"<[^>]+>", "", caption) + "\n-----------------")

        if args.dry_run:
            print(
                f"[DRY-RUN] Would send {'photo' if image_path and not args.no_photo else 'message'}."
            )
            ok = True
        else:
            if image_path and not args.no_photo:
                ok = send_photo_to_telegram(image_path, caption, url_button=url_btn)
            else:
                ok = send_to_telegram(caption, url_button=url_btn)

        if ok:
            # Append to sent_df for next iteration of selection in this run
            sent_row = pd.DataFrame(
                [{col: chosen.get(col) for col in filtered.columns if col in chosen}]
            )
            sent_row["SentAt"] = datetime.utcnow().isoformat()
            sent_df = pd.concat([sent_df, sent_row], ignore_index=True, sort=False)
            sent_this_run.append(chosen)
        else:
            print("[WARN] Sending failed; stopping further sends in this run.")
            break

    # Persist sent history at the end (unless dry-run)
    if not args.dry_run and len(sent_this_run) > 0:
        to_persist = pd.DataFrame(sent_df)
        to_persist.drop_duplicates(subset=["Ids"], keep="last", inplace=True)
        save_dataframe(to_persist, SENT_VACANCIES_CSV, overwrite=True)

    print(f"[INFO] Done. Sent {len(sent_this_run)} message(s).")


# ---------------- Main ----------------


def main():
    args = parse_args()

    if args.reset_sent and os.path.exists(SENT_VACANCIES_CSV):
        os.remove(SENT_VACANCIES_CSV)
        print("[INFO] Reset sent history (file removed).")

    if not args.loop:
        run_job(args)
        return

    # Loop mode: run every day at specified time (local)
    hh, mm = map(int, args.at.strip().split(":"))
    print(f"[INFO] Loop mode enabled. Scheduled daily at {hh:02d}:{mm:02d} (local time).")
    while True:
        now = datetime.now()
        run_at = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if run_at <= now:
            run_at = run_at + timedelta(days=1)
        wait_s = (run_at - now).total_seconds()
        hrs = int(wait_s // 3600)
        mins = int((wait_s % 3600) // 60)
        print(f"[INFO] Next run at {run_at.strftime('%Y-%m-%d %H:%M')}. Sleeping {hrs}h {mins}m…")
        time.sleep(wait_s)
        try:
            run_job(args)
        except Exception as e:
            print(f"[ERROR] Job failed: {e}")
        # continue loop for next day


if __name__ == "__main__":
    main()
