#Nacharbeit., da uns das Datum fehlt im Stages Datensatz

import json
import re
import time
import random
import os
from curl_cffi import requests
from selectolax.parser import HTMLParser

# --- KONFIGURATION ---
RACES = ["tour-de-france", "giro-d-italia", "vuelta-a-espana"]
YEARS = list(range(2005, 2026)) # Bis 2025
OUTPUT_FILE = "grand_tour_stages_with_dates.jsonl"
CHECKPOINT_FILE = "stage_date_checkpoint.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
}

def get_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"race_idx": 0, "year_idx": 0, "stage_idx": 1}

def save_checkpoint(r_idx, y_idx, s_idx):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"race_idx": r_idx, "year_idx": y_idx, "stage_idx": s_idx}, f)

def scrape_stage_with_date(url, race_name, year, stage_nr):
    try:
        resp = requests.get(url, headers=HEADERS, impersonate="chrome120", timeout=30)
        if resp.status_code != 200:
            return None

        tree = HTMLParser(resp.text)
        # Wir suchen im Header/Info-Bereich nach dem Datum
        info_block = tree.css_first("ul.main")
        full_text = tree.body.text()

        # Datum Extraktion (Format auf PCS oft: 19 July 2024)
        # Wir suchen nach dem Muster "Date: 19 July 2024"
        date_match = re.search(r"Date:\s*(\d{1,2}\s+[a-zA-Z]+\s+\d{4})", full_text)
        stage_date = date_match.group(1) if date_match else "n/a"

        # Metadaten Extraktion (wie gehabt)
        def get_val(label):
            pattern = rf"{label}:\s*([^\n\r|]+)"
            match = re.search(pattern, full_text, re.IGNORECASE)
            return match.group(1).strip() if match else "n/a"

        stage_data = {
            "race": race_name,
            "year": year,
            "stage_nr": stage_nr,
            "date": stage_date,
            "url": url,
            "metadata": {
                "distance": get_val("Distance"),
                "vertical_meters": get_val("Vertical meters"),
                "won_how": get_val("Won how"),
                "profile_score": get_val("ProfileScore")
            }
        }

        return stage_data
    except Exception as e:
        print(f"\nFehler bei {url}: {e}")
        return None

# --- MAIN LOOP ---
checkpoint = get_checkpoint()

for r_idx in range(checkpoint["race_idx"], len(RACES)):
    race = RACES[r_idx]
    for y_idx in range(checkpoint["year_idx"] if r_idx == checkpoint["race_idx"] else 0, len(YEARS)):
        year = YEARS[y_idx]

        for s_idx in range(checkpoint["stage_idx"] if (r_idx == checkpoint["race_idx"] and y_idx == checkpoint["year_idx"]) else 1, 22):
            url = f"https://www.procyclingstats.com/race/{race}/{year}/stage-{s_idx}"

            print(f"Scraping Date: {race} {year} - S{s_idx}...", end="\r")
            data = scrape_stage_with_date(url, race, year, s_idx)

            if data:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data) + "\n")

                # Kurze Bestätigung alle 7 Etappen
                if s_idx % 7 == 0:
                    print(f"\n✅ {race} {year} S{s_idx}: Datum gefunden -> {data['date']}")

            save_checkpoint(r_idx, y_idx, s_idx + 1)
            # Kürzere Pause, da wir weniger Daten pro Seite ziehen
            time.sleep(random.uniform(3, 6))

        save_checkpoint(r_idx, y_idx + 1, 1)
    save_checkpoint(r_idx + 1, 0, 1)

print("\n--- Alle Etappendaten inkl. Datum sind im Kasten! ---")
