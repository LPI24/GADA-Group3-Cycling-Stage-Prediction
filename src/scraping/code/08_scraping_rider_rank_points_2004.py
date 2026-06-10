import json
import random
import time
from pathlib import Path

import pandas as pd
from curl_cffi import requests
from selectolax.parser import HTMLParser


# --------------------------------------------------
# Pfade
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "25_cleaned_master_data.pkl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "rider_rank_history.jsonl"
)

print(f"Input : {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")


# --------------------------------------------------
# Daten laden
# --------------------------------------------------

df = pd.read_pickle(INPUT_FILE)

distinct_rider_urls = (
    df["meta_rider_url"]
      .dropna()
      .unique()
      .tolist()
)

print(f"Zu verarbeitende Fahrer: {len(distinct_rider_urls)}")


# --------------------------------------------------
# Parser
# --------------------------------------------------

def extract_ranking_history(html_text):

    html = HTMLParser(html_text)

    rankings = []

    for row in html.css("table tr"):

        cells = row.css("td")

        if len(cells) != 3:
            continue

        try:

            season = int(
                cells[0].text(strip=True)
            )

            points = int(
                cells[1]
                .text(strip=True)
                .replace(",", "")
            )

            rank = int(
                cells[2]
                .text(strip=True)
                .replace(",", "")
            )

            if season < 2004 or season > 2025:
                continue

            rankings.append(
                {
                    "season": season,
                    "points": points,
                    "rank": rank,
                }
            )

        except Exception:
            continue

    return rankings


# --------------------------------------------------
# Bereits gespeicherte Fahrer laden
# --------------------------------------------------

done_riders = set()

if OUTPUT_FILE.exists():

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            try:

                done_riders.add(
                    json.loads(line)["url_name"]
                )

            except Exception:
                pass

print(
    f"Bereits vorhanden: {len(done_riders)} Fahrer"
)


# --------------------------------------------------
# Scraping
# --------------------------------------------------

for idx, rider_url in enumerate(
    distinct_rider_urls,
    start=1
):

    if rider_url in done_riders:
        continue

    try:

        print(
            f"[{idx}/{len(distinct_rider_urls)}] "
            f"Lade {rider_url}"
        )

        url = (
            f"https://www.procyclingstats.com/"
            f"rider/{rider_url}"
        )

        response = requests.get(
            url,
            impersonate="chrome120",
            timeout=30
        )

        if response.status_code != 200:

            print(
                f"  -> HTTP {response.status_code}"
            )

            continue

        rankings = extract_ranking_history(
            response.text
        )

        if len(rankings) == 0:

            print(
                "  -> Keine Ranking-Historie gefunden"
            )

            continue

        with open(
            OUTPUT_FILE,
            "a",
            encoding="utf-8"
        ) as f:

            for ranking in rankings:

                record = {
                    "url_name": rider_url,
                    **ranking
                }

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

        print(
            f"  -> {len(rankings)} Seasons gespeichert"
        )

        time.sleep(
            random.uniform(2, 4)
        )

    except Exception as e:

        print(
            f"Fehler bei {rider_url}: {e}"
        )

        time.sleep(5)

print("\nFertig.")
