import re
from curl_cffi import requests
from selectolax.parser import HTMLParser

def scrape_full_stage_details(url):
    print(f"\n--- Deep Scan Etappe: {url} ---")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}

    try:
        resp = requests.get(url, headers=headers, impersonate="chrome120")
        tree = HTMLParser(resp.text)
        full_text = tree.body.text()

        # 1. ERWEITERTE METADATEN (Regex)
        def get_val(label):
            pattern = rf"{label}:\s*([^\n\r|]+)"
            match = re.search(pattern, full_text, re.IGNORECASE)
            return match.group(1).strip() if match else "n/a"

        stage_meta = {
            "departure": get_val("Departure"),
            "arrival": get_val("Arrival"),
            "distance": get_val("Distance"),
            "vertical_meters": get_val("Vertical meters"),
            "profile_score": get_val("ProfileScore"),
            "won_how": get_val("Won how"),
            "avg_speed": get_val("Avg. speed winner"),
            "race_ranking": get_val("Race ranking")
        }

        # 2. ERGEBNISSE (Top 20 + Zeitabstände)
        results = []
        table = tree.css_first("table.results")

        if table:
            rows = table.css("tbody tr")
            for row in rows[:20]: # Limit auf die ersten 20 Fahrer
                cells = row.css("td")
                rider_link = row.css_first('a[href*="rider/"]')

                if rider_link and len(cells) >= 5:
                    rank = cells[0].text().strip()
                    rider_url = rider_link.attributes.get("href", "").replace("rider/", "")

                    # Zeit-Extraktion: Wir suchen die Zelle mit Zeitformat oder Gap (+ / ,,)
                    # Bei PCS meistens die 6. Spalte (Index 5) oder 7. Spalte
                    time_gap = "n/a"
                    for cell in cells[4:]:
                        c_text = cell.text().strip()
                        if any(char in c_text for char in [':', '+', ',,']):
                            time_val = c_text
                            break

                    results.append({
                        "rank": rank,
                        "rider_url": rider_url,
                        "time_gap": time_val
                    })

        # 3. KONTROLLAUSGABE
        print(f"📊 Etappen-Steckbrief:")
        for key, val in stage_meta.items():
            print(f"   {key.replace('_', ' ').title()}: {val}")

        print(f"\n⏱️ Top 20 Ergebnisse (Auszug):")
        for r in results:
            print(f"   {r['rank'].split()[-1]:>2}. {r['rider_url']:<25} | Gap: {r['time_gap']}")

        return {"metadata": stage_meta, "results": results}

    except Exception as e:
        print(f"❌ Fehler: {str(e)}")

# Testlauf
scrape_full_stage_details("https://www.procyclingstats.com/race/tour-de-france/2025/stage-1")
