import time
import json
import os
import random
from curl_cffi import requests
from procyclingstats import Rider, Scraper
from selectolax.parser import HTMLParser

# --- KONFIGURATION ---
STAGES_INPUT_FILE = "grand_tour_stage_results.jsonl"  # Datei vom Pi
FINAL_DATA_FILE = "riders_full_biometrics1.jsonl"     # Dein Biometrie-Speicher
LOG_FILE = "scraping_process_pc.log"

# --- ULTIMATE MONKEY PATCH (Bleibt gleich) ---
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

# --- HAUPTFUNKTION ---

def run_direct_rider_scrape():
    # 1. Vorhandene Profile laden (Checkpoint)
    done_riders = set()
    if os.path.exists(FINAL_DATA_FILE):
        with open(FINAL_DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_riders.add(json.loads(line).get("url_name"))
                except: continue
    log(f"Bereits {len(done_riders)} Profile lokal vorhanden.")

    # 2. Einzigartige Rider-URLs aus den Stage Results extrahieren
    log(f"Extrahiere Fahrer-Namen aus {STAGES_INPUT_FILE}...")
    target_riders = set()
    if not os.path.exists(STAGES_INPUT_FILE):
        log("FEHLER: Stage Results Datei nicht gefunden!")
        return

    with open(STAGES_INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                stage = json.loads(line)
                for res in stage.get("results", []):
                    r_url = res.get("rider_url")
                    if r_url and r_url not in done_riders:
                        target_riders.add(r_url)
            except: continue

    log(f"Gefundene neue Fahrer zum Scrapen: {len(target_riders)}")

    # 3. Scraping-Loop für die neuen Fahrer
    for r_url in target_riders:
        log(f"Extrahiere Profil: {r_url}...")
        try:
            full_path = f"rider/{r_url}"
            url = f"https://www.procyclingstats.com/{full_path}"

            resp = requests.get(
                url,
                impersonate="chrome120",
                timeout=30
            )

            if resp.status_code == 200 and resp.text:
                if "Just a moment" in resp.text:
                    log(f"⚠️ Cloudflare! Pause 60s...")
                    time.sleep(60)
                    continue

                # PCS-Library Patching
                rider = Rider(full_path, html=resp.text)
                rider._html_str = resp.text
                rider.update_html()

                data = rider.parse()

                if data:
                    data["url_name"] = r_url
                    # Speichern (Append-Mode)
                    with open(FINAL_DATA_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(data) + "\n")
                    log(f"✅ {r_url} gesichert.")
                else:
                    log(f"⚠️ Parsing leer für {r_url}")
            else:
                log(f"⚠️ Status {resp.status_code} bei {r_url}")

            # Politeness Pause
            time.sleep(random.uniform(7, 12))

        except Exception as e:
            log(f"❌ Fehler bei {r_url}: {str(e)}")
            time.sleep(10)

if __name__ == "__main__":
    run_direct_rider_scrape()
