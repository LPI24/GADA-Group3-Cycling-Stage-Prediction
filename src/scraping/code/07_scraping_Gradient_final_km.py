import json
import re
import time
import random
import os
from curl_cffi import requests
from selectolax.parser import HTMLParser

# --- KONFIGURATION ---
RACES = ["tour-de-france", "giro-d-italia", "vuelta-a-espana"]
YEARS = list(range(2005, 2026)) # Bis 2025 voll abgedeckt
OUTPUT_FILE = "gradient_final_km_only.jsonl"
CHECKPOINT_FILE = "gradient_checkpoint.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
}

# --- FUNKTIONEN ---

def get_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"race_idx": 0, "year_idx": 0, "stage_idx": 1}

def save_checkpoint(r_idx, y_idx, s_idx):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"race_idx": r_idx, "year_idx": y_idx, "stage_idx": s_idx}, f)

def scrape_gradient_only(url, race_name, year, stage_nr):
    try:
        resp = requests.get(url, headers=HEADERS, impersonate="chrome120", timeout=30)
        if resp.status_code != 200:
            return None

        tree = HTMLParser(resp.text)
        full_text = tree.body.text()

        # Regex sucht nach "Gradient final km:" und fängt alles bis zum nächsten Zeilenumbruch oder Trenner
        pattern = r"Gradient final km:\s*([^\n\r|]+)"
        match = re.search(pattern, full_text, re.IGNORECASE)
        gradient_val = match.group(1).strip() if match else "0.0%" # Wenn nicht angegeben, meist flach (0.0%)

        # Schlankes Datenpaket für den späteren Join
        return {
            "race": race_name,
            "year": year,
            "stage_nr": stage_nr,
            "url": url,
            "gradient_final_km": gradient_val
        }
    except Exception as e:
        print(f" ❌ Fehler bei {url}: {e}")
        return None

# --- MAIN LOOP ---

checkpoint = get_checkpoint()

for r_idx in range(checkpoint["race_idx"], len(RACES)):
    race = RACES[r_idx]
    for y_idx in range(checkpoint["year_idx"] if r_idx == checkpoint["race_idx"] else 0, len(YEARS)):
        year = YEARS[y_idx]

        for s_idx in range(checkpoint["stage_idx"] if (r_idx == checkpoint["race_idx"] and y_idx == checkpoint["year_idx"]) else 1, 22):
            url = f"https://www.procyclingstats.com/race/{race}/{year}/stage-{s_idx}"

            print(f"Scraping: {race} {year} - Stage {s_idx}...", end="\r")
            data = scrape_gradient_only(url, race, year, s_idx)

            if data:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data) + "\n")

                # Kurzes visuelles Feedback in der Konsole
                print(f"📈 {race.upper()} {year} S{s_idx:<2} -> Gradient Final KM: {data['gradient_final_km']}")

            save_checkpoint(r_idx, y_idx, s_idx + 1)
            time.sleep(random.uniform(5, 10)) # Da wir keine Tabellen parsen, können wir das Delay minimal senken (5-10s)

        save_checkpoint(r_idx, y_idx + 1, 1)
    save_checkpoint(r_idx + 1, 0, 1)

print("\n--- Scraper erfolgreich beendet. Datei bereit für den Pandas-Join! ---")
