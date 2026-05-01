import time
import json
import os
import random
from curl_cffi import requests
from procyclingstats import Rider, Scraper
from selectolax.parser import HTMLParser

# --- KONFIGURATION (Alle Dateien im selben Ordner) ---
STAGES_FILE = "grand_tour_stage_results.jsonl"
BIOMETRICS_FILE = "riders_full_biometrics.jsonl"
LOG_FILE = "scraper_output.log"

# --- ULTIMATE MONKEY PATCH ---
def dummy_check(self): return
Scraper._check_cloudflare = dummy_check
Scraper._make_request = lambda self, url: None

def forced_update_html(self):
    if hasattr(self, "_html_str") and self._html_str:
        self._html = HTMLParser(self._html_str)
    else:
        self._html = HTMLParser("<html></html>")

Scraper.update_html = forced_update_html

def log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")

def run_pi_rider_scraper():
    # 1. Schritt: Vorhandene Profile laden
    existing_profiles = set()
    if os.path.exists(BIOMETRICS_FILE):
        log("Lade bereits vorhandene Profile aus riders_full_biometrics.jsonl...")
        with open(BIOMETRICS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing_profiles.add(json.loads(line).get("url_name"))
                except: continue
    log(f"Bereits vorhanden: {len(existing_profiles)} Profile.")

    # 2. Schritt: Neue Fahrer aus Stage Results extrahieren
    if not os.path.exists(STAGES_FILE):
        log(f"FEHLER: {STAGES_FILE} nicht gefunden! Scraper stoppt.")
        return

    log(f"Analysiere {STAGES_FILE} nach neuen Fahrern...")
    needed_riders = set()
    with open(STAGES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                stage = json.loads(line)
                for res in stage.get("results", []):
                    r_url = res.get("rider_url")
                    if r_url and r_url not in existing_profiles:
                        needed_riders.add(r_url)
            except: continue

    log(f"Gefunden: {len(needed_riders)} neue Fahrer zum Scrapen.")

    if len(needed_riders) == 0:
        log("Keine neuen Fahrer gefunden. Alles auf dem neuesten Stand!")
        return

    # 3. Schritt: Scraping-Loop
    for r_url in needed_riders:
        log(f"Scrape Profil: {r_url}...")
        try:
            full_path = f"rider/{r_url}"
            url = f"https://www.procyclingstats.com/{full_path}"

            resp = requests.get(url, impersonate="chrome120", timeout=30)

            if resp.status_code == 200 and "Just a moment" not in resp.text:
                rider = Rider(full_path, html=resp.text)
                rider._html_str = resp.text
                rider.update_html()

                data = rider.parse()
                if data:
                    data["url_name"] = r_url
                    # Append-Mode: Fügt den neuen Fahrer unten an die Datei an
                    with open(BIOMETRICS_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(data) + "\n")
                    log(f"✅ {r_url} gesichert (inkl. Team History).")

                # Pause für den Pi (zwischen 7 und 12 Sekunden)
                time.sleep(random.uniform(7, 12))

            elif "Just a moment" in resp.text:
                log("⚠️ Cloudflare erkannt! Pause 60s...")
                time.sleep(60)
            else:
                log(f"⚠️ Status {resp.status_code} bei {r_url}. Überspringe...")

        except Exception as e:
            log(f"❌ Fehler bei {r_url}: {str(e)}")
            time.sleep(10)

if __name__ == "__main__":
    run_pi_rider_scraper()
