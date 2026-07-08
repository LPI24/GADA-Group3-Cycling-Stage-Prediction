# _02_stage_scraper.py (FINALE VERSION MIT ERGEBNIS-EXTRAKTION)

import re
import pandas as pd
from curl_cffi import requests
from selectolax.parser import HTMLParser

def scrape_pcs_stage_clean(url, user_inputs):
    """
    Scrapt eine PCS-Etappenseite und liefert exakt zwei DataFrames zurück.
    Extrahiert automatisch den Rang im Ziel, falls die Etappe bereits gelaufen ist.
    """
    print(f"\n--- Starte Live-Etappen-Scan: {url} ---")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}

    try:
        # 1. HTML sicher abrufen (TLS-Fingerprint-Impersonation)
        resp = requests.get(url, headers=headers, impersonate="chrome120", timeout=15)
        if resp.status_code != 200:
            print(f"Fehler: PCS antwortet mit Status-Code {resp.status_code}")
            return pd.DataFrame(), pd.DataFrame()

        tree = HTMLParser(resp.text)
        full_text = tree.body.text()

        # Helper-Funktion um Roh-Texte via Regex aus dem Gesamttext zu fischen
        def get_raw_value(label):
            pattern = rf"{label}:\s*([^\n\r|]+)"
            match = re.search(pattern, full_text, re.IGNORECASE)
            return match.group(1).strip() if match else None

        # ==================================================================
        # DATAFRAME 1: METADATEN FÜR DAS MODELL
        # ==================================================================
        raw_dist = get_raw_value("Distance")
        distance_val = float(re.search(r'([\d\.]+)', raw_dist).group(1)) if raw_dist else None

        raw_vm = get_raw_value("Vertical meters")
        vm_val = int(re.search(r'(\d+)', raw_vm).group(1)) if raw_vm else None

        raw_grad = get_raw_value("Gradient final km")
        grad_val = float(re.search(r'([\d\.\-]+)', raw_grad).group(1)) if raw_grad else None

        date_val = get_raw_value("Date")

        race_input = user_inputs.get("race_slug", "")
        race_clean = "tdf" if "tour" in race_input else ("giro" if "giro" in race_input else "vuelta")

        stage_meta_data = {
            "stage_url": url,
            "race": race_clean,
            "date": date_val,
            "distance": distance_val,
            "vertical_meters": vm_val,
            "stage_nr": int(user_inputs.get("stage_num", 0)),
            "gradient_final_km": grad_val
        }
        df_stage_meta = pd.DataFrame([stage_meta_data])

        # ==================================================================
        # DATAFRAME 2: DIE STARTLISTE & REALER RANG
        # ==================================================================
        riders_list = []
        table = tree.css_first("table.results")

        if table:
            rows = table.css("tbody tr")
            for row in rows:
                cells = row.css("td")
                rider_link = row.css_first('a[href*="rider/"]')

                # Valide Ergebnis-Zeile prüfen
                if rider_link and len(cells) >= 5:
                    raw_rank = cells[0].text().strip()

                    # Säuberung des Rangs (z.B. "1", "DNF", "" bei reinen Startlisten)
                    try:
                        # Extrahiere nur Ziffern (falls Punkte oder Zusätze wie "1st" enthalten sind)
                        digit_match = re.search(r'(\d+)', raw_rank)
                        rank_val = int(digit_match.group(1)) if digit_match else None
                    except ValueError:
                        rank_val = None

                    # URL extrahieren & säubern
                    raw_href = rider_link.attributes.get("href", "")
                    rider_url_clean = raw_href.split('/')[-1] if '/' in raw_href else raw_href
                    rider_name = rider_link.text().strip()

                    riders_list.append({
                        "rank": rank_val,  # Enthält den echten Platz (Integer) oder None
                        "rider_url": rider_url_clean,
                        "rider_name": rider_name
                    })

        df_startlist = pd.DataFrame(riders_list)

        print("➔ Metadaten-Extraktion abgeschlossen.")
        print(f"➔ Startliste geladen: {len(df_startlist)} Fahrer gefunden.")

        return df_startlist, df_stage_meta

    except Exception as e:
        print(f"Schwerer Fehler im Scraper-Modul: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()
