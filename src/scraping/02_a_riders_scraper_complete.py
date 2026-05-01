import time
import json
import os
import random
from curl_cffi import requests
from procyclingstats import Rider, Scraper
from selectolax.parser import HTMLParser

# --- KONFIGURATION (Dynamische Pfadberechnung) ---
# Holt den absoluten Pfad zum Ordner, in dem dieses Skript liegt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Von src/scraping/code zwei Ebenen hoch zu src, dann zu data/raw
# Sicherer Weg:
BASE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "raw"))
LOG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "logs"))

STAGES_FILE = os.path.join(BASE_PATH, "V1_grand_tour_stage_results.jsonl")
BIOMETRICS_FILE = os.path.join(BASE_PATH, "riders_full_biometrics.jsonl")
LOG_FILE = os.path.join(LOG_DIR, "rider_scraping.log")

# Kleiner Debug-Print beim Start, damit du siehst, wo er sucht:
print(f"Suche Stages in: {STAGES_FILE}")
print(f"Suche Biometrics in: {BIOMETRICS_FILE}")

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
    if not os.path.exists("../../logs"): os.makedirs("../../logs")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")

def run_smart_scraper():
    # 1. Schritt: Vorhandene Profile laden (Was haben wir schon?)
    existing_profiles = set()
    if os.path.exists(BIOMETRICS_FILE):
        log("Lade vorhandene Biometrie-Profile...")
        with open(BIOMETRICS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing_profiles.add(json.loads(line).get("url_name"))
                except: continue
    log(f"Gefundene Profile: {len(existing_profiles)}")

    # 2. Schritt: Fehlende Fahrer aus den Stage Results identifizieren
    log(f"Abgleich mit {STAGES_FILE}...")
    needed_riders = set()
    if os.path.exists(STAGES_FILE):
        with open(STAGES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    stage = json.loads(line)
                    for res in stage.get("results", []):
                        r_url = res.get("rider_url")
                        if r_url and r_url not in existing_profiles:
                            needed_riders.add(r_url)
                except: continue

    log(f"Ergebnis: {len(needed_riders)} Fahrer fehlen noch und werden jetzt gescrapt.")

    # 3. Schritt: Nur die fehlenden Fahrer scrapen
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
                    # Sofort in die Datei speichern (Append)
                    with open(BIOMETRICS_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(data) + "\n")
                    log(f"✅ {r_url} erfolgreich hinzugefügt.")

                # Politeness Pause (Wichtig!)
                time.sleep(random.uniform(7, 12))

            elif "Just a moment" in resp.text:
                log("⚠️ Cloudflare Challenge erkannt! Pause 60s...")
                time.sleep(60)
            else:
                log(f"❌ Fehler {resp.status_code} bei {r_url}")

        except Exception as e:
            log(f"❌ Schwerer Fehler bei {r_url}: {str(e)}")
            time.sleep(5)

if __name__ == "__main__":
    run_smart_scraper()
