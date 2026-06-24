# _05_scraper_missing_lags.py
import re
import pandas as pd
from curl_cffi import requests
from selectolax.parser import HTMLParser
from database_manager import CyclingDatabase


# Wir importieren deine bewährten, robusten Extraktions-Funktionen aus Schritt 4!
from _04_scraper_missing_drivers_profiles import (
    clean_node_text,
    extract_current_team_and_tier,
    extract_lag_points_and_rank
)

def scrape_and_save_missing_lags(df_missing_lags, df_stage_meta):
    """
    Schritt _05_: Ergänzt für bereits in r_master bekannte Fahrer
    die fehlenden historischen Leistungsdaten (Lags) für das Jahr - 1.
    """
    if df_missing_lags.empty:
        print("Keine fehlenden Saison-Lags für bestehende Fahrer. Schritt _05_ wird übersprungen.")
        return

    # Zieljahr für die Lags bestimmen (Etappenjahr - 1) -> Bei Etappe 2026 ist das 2025
    date_str = str(df_stage_meta["date"].iloc[0])
    try:
        stage_year = int(date_str.split()[-1])
    except Exception:
        stage_year = 2026
    target_lag_year = stage_year - 1

    print("\n==================================================================")
    print(f"UPDATE SAISON-LAGS FÜR BESTEHENDE FAHRER ({target_lag_year}) (_05_)")
    print(f"{len(df_missing_lags)} Fahrer werden auf das Vorjahr überprüft.")
    print("==================================================================")

    db = CyclingDatabase(db_path="../../data/databases/cycling_production.db")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    conn = db.get_connection()
    cursor = conn.cursor()

    try:
        for _, row in df_missing_lags.iterrows():
            url_slug = row["rider_url"]
            display_name = row["rider_name"]

            full_url = f"https://www.procyclingstats.com/rider/{url_slug}"
            print(f"\n--->Scrape historische Lags ({target_lag_year}): {display_name}")

            try:
                resp = requests.get(full_url, headers=headers, impersonate="chrome120", timeout=12)
            except Exception as e:
                print(f"--Request-Fehler für {url_slug}: {e}")
                continue

            if resp.status_code != 200:
                print(f"--Fehler beim Laden für {url_slug}. Status: {resp.status_code}")
                continue

            tree = HTMLParser(resp.text)

            # ==================================================================
            # 1. TEAM & TIER EXTRAHIEREN (Inkl. deiner CLUB-Kick-Logik!)
            # ==================================================================
            current_team, team_tier = extract_current_team_and_tier(tree, target_lag_year)

            if team_tier == "SKIP_CLUB":
                print(f"--Fahrer {display_name} wird ignoriert (Tier: CLUB). Kein DB-Update.")
                continue

            # ==================================================================
            # 2. PUNKTE & RANG EXTRAHIEREN
            # ==================================================================
            lag_rider_points_season, lag_rider_rank_season = extract_lag_points_and_rank(tree, target_lag_year)

            # ==================================================================
            # 3. IN LAGS_HISTORICAL SCHREIBEN (Erzeugt das 2025er Jahr bei einer Etappe aus 2026)
            # ==================================================================
            cursor.execute(
                """
                INSERT INTO lags_historical (
                    rider_url, season_year, current_team, team_tier, lag_rider_points_season, lag_rider_rank_season
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(rider_url, season_year) DO UPDATE SET
                    current_team = excluded.current_team,
                    team_tier = excluded.team_tier,
                    lag_rider_points_season = excluded.lag_rider_points_season,
                    lag_rider_rank_season = excluded.lag_rider_rank_season
                """,
                (url_slug, target_lag_year, current_team, team_tier, lag_rider_points_season, lag_rider_rank_season),
            )

            print(f"---Lag-Update komplett! Team: {current_team} | Tier: {team_tier} | Punkte: {lag_rider_points_season} | Rang: {lag_rider_rank_season}")

        conn.commit()
        print("\n==================================================================")
        print("ALLE FEHLENDEN SAISON-LAGS ERFOLGREICH IN DIE DB INTEGRIERT!")
        print("==================================================================")

    except Exception as e:
        print(f"Schwerer Fehler in Schritt _05_: {e}")
        conn.rollback()
    finally:
        conn.close()
