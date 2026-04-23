import time
import json
import os
import random
from curl_cffi import requests
from procyclingstats import Rider, Scraper
from selectolax.parser import HTMLParser

# --- ULTIMATE MONKEY PATCH ---
# Wir binden die Hände der Library, damit sie nicht selbst ins Netz geht
def dummy_check(self):
    return

# Wir verhindern jegliche Eigeninitiative beim Laden
Scraper._check_cloudflare = dummy_check
Scraper._make_request = lambda self, url: None

# Wir zwingen die Library, unseren HTML-String direkt zu parsen
def forced_update_html(self):
    if hasattr(self, "_html_str") and self._html_str:
        self._html = HTMLParser(self._html_str)
    else:
        # Falls kein HTML da ist, verhindern wir den Absturz durch einen leeren Parser
        self._html = HTMLParser("<html></html>")

Scraper.update_html = forced_update_html
# -----------------------------

# --- KONFIGURATION ---
INDEX_FILE = "master_riders_index.jsonl"
FINAL_DATA_FILE = "riders_full_biometrics.jsonl"
LOG_FILE = "scraping_process_pc.log"

def log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")

def run_stage_2_pc(rider_index):
    log(f"Starte Detail-Parsing für {len(rider_index)} Fahrer auf dem PC...")

    # Checkpoint-Logik
    done_riders = set()
    if os.path.exists(FINAL_DATA_FILE):
        with open(FINAL_DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_riders.add(json.loads(line).get("url_name"))
                except: continue

    for r_url in rider_index.keys():
        if r_url in done_riders:
            continue

        log(f"Extrahiere Profil: {r_url}...")
        try:
            full_path = f"rider/{r_url}"
            url = f"https://www.procyclingstats.com/{full_path}"

            # Der sichere Request über curl_cffi
            resp = requests.get(
                url,
                impersonate="chrome120",
                timeout=30
            )

            if resp.status_code == 200 and resp.text:
                if "Just a moment" in resp.text:
                    log(f"⚠️ Cloudflare Challenge am PC! Bitte kurz im Browser PCS öffnen. Pause 60s...")
                    time.sleep(60)
                    continue

                # Rider Objekt initialisieren
                # Wir übergeben das HTML und setzen das Attribut für unseren Patch
                rider = Rider(full_path, html=resp.text)
                rider._html_str = resp.text
                rider.update_html() # Triggert unseren forced_update_html Patch

                # Jetzt das eigentliche Parsing
                data = rider.parse()

                if data:
                    data["url_name"] = r_url
                    data["grand_tour_history"] = rider_index[r_url]["history"]

                    # Speichern
                    with open(FINAL_DATA_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(data) + "\n")
                    log(f"✅ {r_url} gesichert.")
                else:
                    log(f"⚠️ Parsing-Ergebnis leer für {r_url}")
            else:
                log(f"⚠️ Status {resp.status_code} bei {r_url}")

            # Menschliche Pause am PC
            time.sleep(random.uniform(7, 12))

        except Exception as e:
            log(f"❌ Schwerer Fehler bei {r_url}: {str(e)}")
            time.sleep(15)

if __name__ == "__main__":
    if not os.path.exists(INDEX_FILE):
        log(f"FEHLER: {INDEX_FILE} nicht gefunden! Bitte vom Pi kopieren.")
    else:
        # Index laden
        r_index = {}
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    r_index[item["url_name"]] = item
                except: continue

        log(f"{len(r_index)} Fahrer im Index geladen.")
        run_stage_2_pc(r_index)
