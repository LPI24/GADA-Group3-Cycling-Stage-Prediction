# _04_rider_scraper.py
import re
from datetime import datetime
import pandas as pd
from curl_cffi import requests
from selectolax.parser import HTMLParser
from database_manager import CyclingDatabase

def scrape_and_save_missing_profiles(df_missing_master, df_stage_meta):
    """
    Scrapt für alle komplett unbekannten Fahrer sowohl die biometrischen Stammdaten
    als auch die historischen Leistungs-Lags des Vorjahres in einem einzigen Durchlauf.
    Setzt Punkte auf 0.0 und Rang auf 9999, falls für das Vorjahr keine Daten existieren.
    """
    if df_missing_master.empty:
        print("Keine komplett fehlenden Fahrer-Profile. Schritt _04_ wird übersprungen.")
        return

    # Zieljahr für die Lags bestimmen (Etappenjahr - 1)
    date_str = str(df_stage_meta['date'].iloc[0])
    try:
        stage_year = int(date_str.split()[-1])
    except:
        stage_year = 2026
    target_lag_year = stage_year - 1

    print("\n==================================================================")
    print(f"DEEP-SCAN FEHLENDER PROFILE & LAGS ({target_lag_year}) (_04_)")
    print(f"- {len(df_missing_master)} Profile werden vollständig erfasst.")
    print("==================================================================")

    db = CyclingDatabase(db_path="../../data/databases/cycling_production.db")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}

    conn = db.get_connection()
    cursor = conn.cursor()

    try:
        for _, row in df_missing_master.iterrows():
            url_slug = row['rider_url']
            display_name = row['rider_name']

            full_url = f"https://www.procyclingstats.com/rider/{url_slug}"
            print(f"\nScrape Vollprofil: {display_name}...")

            resp = requests.get(full_url, headers=headers, impersonate="chrome120", timeout=12)
            if resp.status_code != 200:
                print(f"   ⚠️ Fehler: Konnte Seite für {url_slug} nicht laden.")
                continue

            tree = HTMLParser(resp.text)

            # Info-Block als Text-Basis sichern
            info_cont = tree.css_first("div.rdr-info-cont")
            info_str = info_cont.text() if info_cont else ""

            # ==================================================================
            # TEIL 1: STAMMDATEN PARSEN (r_master)
            # ==================================================================
            nationality = "UNKNOWN"
            nat_link = tree.css_first("a[href^='nation/']")
            if nat_link:
                nationality = nat_link.text().strip()[:3].upper()

            weight_match = re.search(r"Weight:\s*(\d+)", info_str, re.IGNORECASE)
            weight = float(weight_match.group(1)) if weight_match else None

            height_match = re.search(r"Height:\s*([\d\.]+)", info_str, re.IGNORECASE)
            height = float(height_match.group(1)) if height_match else None

            bmi = round(weight / (height ** 2), 2) if height and weight and height > 0 else None

            birthdate_str = None
            for li in tree.css("div.rdr-info-cont ul.list li"):
                if "Date of birth:" in li.text():
                    date_parts = [div.text().strip() for div in li.css("div") if div.text().strip()][1:]
                    if len(date_parts) >= 3:
                        day = re.sub(r"\D", "", date_parts[0])
                        month = date_parts[1]
                        year = date_parts[2]
                        try:
                            birthdate_obj = datetime.strptime(f"{day} {month} {year}", "%d %B %Y")
                            birthdate_str = birthdate_obj.strftime("%Y-%m-%d")
                        except:
                            birthdate_str = f"{year}-{month}-{day}"
                    break

            # ==================================================================
            # TEIL 2: SAISON-LAGS PARSEN (lags_historical)
            # ==================================================================
            current_team = "None / No Team"
            team_tier = "UNKNOWN"

            teams_ul = tree.css_first("ul.rdr-teams2")
            if teams_ul:
                for li in teams_ul.css("li"):
                    season_div = li.css_first("div.season")
                    name_div = li.css_first("div.name") or li.css_first("div.name2")

                    if season_div and name_div and str(target_lag_year) in season_div.text():
                        team_link = name_div.css_first("a")
                        if team_link:
                            current_team = team_link.text().strip()
                        tier_match = re.search(r"\((WT|PRT|CT|CLUB)\)", name_div.text())
                        if tier_match:
                            team_tier = tier_match.group(1)
                        break

            # Standard-Füllwerte setzen, falls der Fahrer im Vorjahr inaktiv/nicht gelistet war
            lag_rider_points_season = 0.0
            lag_rider_rank_season = 9999

            ranking_table = tree.css_first("table.basic.small")
            if ranking_table:
                for tr in ranking_table.css("tbody tr"):
                    tds = tr.css("td")
                    if len(tds) >= 3:
                        year_link = tds[0].css_first("a")
                        if year_link and year_link.text().strip() == str(target_lag_year):
                            points_div = tds[1].css_first("div.title")
                            if points_div:
                                lag_rider_points_season = float(points_div.text().strip())

                            rank_text = tds[2].text().strip()
                            if rank_text.isdigit():
                                lag_rider_rank_season = int(rank_text)
                            break

            # ==================================================================
            # TEIL 3: DIREKT IN DIE DB SCHREIBEN
            # ==================================================================
            cursor.execute('''
                INSERT INTO r_master (rider_url, meta_name, meta_url_name, height, weight, rider_bmi, nationality, birthdate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rider_url) DO UPDATE SET
                    height=excluded.height, weight=excluded.weight, rider_bmi=excluded.rider_bmi,
                    nationality=excluded.nationality, birthdate=excluded.birthdate
            ''', (url_slug, display_name, url_slug, height, weight, bmi, nationality, birthdate_str))

            cursor.execute('''
                INSERT INTO lags_historical (
                    rider_url, season_year, current_team, team_tier,
                    lag_rider_points_season, lag_rider_rank_season
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(rider_url, season_year) DO UPDATE SET
                    current_team=excluded.current_team, team_tier=excluded.team_tier,
                    lag_rider_points_season=excluded.lag_rider_points_season, lag_rider_rank_season=excluded.lag_rider_rank_season
            ''', (url_slug, target_lag_year, current_team, team_tier, lag_rider_points_season, lag_rider_rank_season))

            print(f"- DB-Eintrag komplett! BMI: {bmi} | Team: {current_team} | Punkte {target_lag_year}: {lag_rider_points_season} | Rang: {lag_rider_rank_season}")

        conn.commit()
        print("\n==================================================================")
        print("FEHLENDE PROFILE INKLUSIVE IHRER LAGS ERFOLGREICH GESPEICHERT!")
        print("==================================================================")

    except Exception as e:
        print(f"Fehler bei der Erfassung der Profildaten: {e}")
        conn.rollback()
    finally:
        conn.close()
