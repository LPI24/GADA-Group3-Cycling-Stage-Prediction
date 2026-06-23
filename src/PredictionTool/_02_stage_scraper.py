# stage_scraper.py
import re
import pandas as pd
from curl_cffi import requests
from selectolax.parser import HTMLParser

def scrape_pcs_stage_clean(url, user_inputs):
    """
    Scrapt eine PCS-Etappenseite und liefert exakt zwei DataFrames zurück.

    Inputs:
    - url: Die zu scrapende PCS-Etappen-URL
    - user_inputs: Dictionary aus dem input_handler (enthält race_slug und stage_num)

    Outputs:
    - df_startlist: DataFrame mit ['rank', 'rider_url', 'rider_name']
    - df_stage_meta: DataFrame mit ['stage_url', 'race', 'date', 'distance', 'vertical_meters', 'stage_nr', 'gradient_final_km']
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

        # Distanz parsen (z.B. "221 km" -> 221.0)
        raw_dist = get_raw_value("Distance")
        distance_val = float(re.search(r'([\d\.]+)', raw_dist).group(1)) if raw_dist else None

        # Höhenmeter parsen (z.B. "2348" -> 2348)
        raw_vm = get_raw_value("Vertical meters")
        vm_val = int(re.search(r'(\d+)', raw_vm).group(1)) if raw_vm else None

        # Letzter Kilometer Steigung (z.B. "1.0%" -> 1.0)
        raw_grad = get_raw_value("Gradient final km")
        grad_val = float(re.search(r'([\d\.\-]+)', raw_grad).group(1)) if raw_grad else None

        # Datum holen (z.B. "09 May 2026")
        date_val = get_raw_value("Date")

        # Welches Rennen (Kürzel mappen für tdf, giro, vuelta)
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
        # DATAFRAME 2: DIE STARTLISTE
        # ==================================================================
        riders_list = []
        table = tree.css_first("table.results")

        if table:
            rows = table.css("tbody tr")
            for row in rows:
                cells = row.css("td")
                rider_link = row.css_first('a[href*="rider/"]')

                # Sicherstellen, dass es eine valide Zeile mit Fahrerlink ist
                if rider_link and len(cells) >= 8:
                    rank = cells[0].text().strip()

                    # URL extrahieren & säubern: 'rider/florian-stork' -> 'florian-stork'
                    raw_href = rider_link.attributes.get("href", "")
                    rider_url_clean = raw_href.split('/')[-1] if '/' in raw_href else raw_href

                    rider_name = rider_link.text().strip()

                    riders_list.append({
                        "rank": rank,
                        "rider_url": rider_url_clean,
                        "rider_name": rider_name
                    })

        df_startlist = pd.DataFrame(riders_list)

        # Schnelle Terminal-Kontrolle für den User
        print("➔ Metadaten-Extraktion abgeschlossen.")
        print(f"➔ Startliste geladen: {len(df_startlist)} Fahrer gefunden.")

        return df_startlist, df_stage_meta

    except Exception as e:
        print(f"Schwerer Fehler im Scraper-Modul: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()
