import json
import re
import time
import random
import os
from curl_cffi import requests
from selectolax.parser import HTMLParser

# --- KONFIGURATION ---
RACES = ["tour-de-france", "giro-d-italia", "vuelta-a-espana"]
YEARS = list(range(2005, 2026))
OUTPUT_FILE = "grand_tour_stage_results.jsonl"
CHECKPOINT_FILE = "stage_checkpoint.json"

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

def scrape_stage(url, race_name, year):
    try:
        resp = requests.get(url, headers=HEADERS, impersonate="chrome120", timeout=30)
        if resp.status_code != 200:
            return None

        tree = HTMLParser(resp.text)
        full_text = tree.body.text()

        # Metadaten Extraktion
        def get_val(label):
            pattern = rf"{label}:\s*([^\n\r|]+)"
            match = re.search(pattern, full_text, re.IGNORECASE)
            return match.group(1).strip() if match else "n/a"

        stage_data = {
            "race": race_name,
            "year": year,
            "url": url,
            "metadata": {
                "departure": get_val("Departure"),
                "arrival": get_val("Arrival"),
                "distance": get_val("Distance"),
                "vertical_meters": get_val("Vertical meters"),
                "profile_score": get_val("ProfileScore"),
                "won_how": get_val("Won how"),
                "avg_speed": get_val("Avg. speed winner"),
                "race_ranking": get_val("Race ranking")
            },
            "results": []
        }

        # Ergebnisse
        table = tree.css_first("table.results")
        if table:
            rows = table.css("tbody tr")
            for row in rows:
                cells = row.css("td")
                rider_link = row.css_first('a[href*="rider/"]')
                if rider_link and len(cells) >= 5:
                    rank = cells[0].text().strip().split()[-1] # Säubert "1 " zu "1"
                    rider_url = rider_link.attributes.get("href", "").replace("rider/", "")

                    time_val = "n/a"
                    for cell in cells[4:]:
                        c_text = cell.text().strip()
                        if any(char in c_text for char in [':', '+', ',,']):
                            time_val = c_text
                            break

                    stage_data["results"].append({
                        "rank": rank,
                        "rider_url": rider_url,
                        "time_gap": time_val
                    })
        return stage_data
    except Exception as e:
        print(f" Fehler bei {url}: {e}")
        return None

# --- MAIN LOOP ---

checkpoint = get_checkpoint()

for r_idx in range(checkpoint["race_idx"], len(RACES)):
    race = RACES[r_idx]
    for y_idx in range(checkpoint["year_idx"] if r_idx == checkpoint["race_idx"] else 0, len(YEARS)):
        year = YEARS[y_idx]

        # Jede Grand Tour hat 21 Etappen
        for s_idx in range(checkpoint["stage_idx"] if (r_idx == checkpoint["race_idx"] and y_idx == checkpoint["year_idx"]) else 1, 22):
            url = f"https://www.procyclingstats.com/race/{race}/{year}/stage-{s_idx}"

            print(f"Scraping: {race} {year} - Stage {s_idx}...", end="\r")
            data = scrape_stage(url, race, year)

            if data and data["results"]:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data) + "\n")

                # Konsolen-Struktur wie gewünscht ausgeben (alle 5 Etappen zur Kontrolle)
                if s_idx % 5 == 1:
                    print(f"\n--- Etappen-Steckbrief: {race.upper()} {year} S{s_idx} ---")
                    for k, v in data["metadata"].items():
                        print(f"   {k.replace('_',' ').title()}: {v}")
                    print(f"⏱️ Top 5 Ergebnisse:")
                    for r in data["results"][:5]:
                        print(f"   {r['rank']}. {r['rider_url']:<20} | Gap: {r['time_gap']}")

            save_checkpoint(r_idx, y_idx, s_idx + 1)

            # WICHTIG: Pi-Sicherheitspause
            time.sleep(random.uniform(8, 15))

        save_checkpoint(r_idx, y_idx + 1, 1)
    save_checkpoint(r_idx + 1, 0, 1)

print("\n--- Programm fertig. ---")
