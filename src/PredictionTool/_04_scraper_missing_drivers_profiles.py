# _04_scraper_missing_drivers_profiles.py
import re
from datetime import datetime
import pandas as pd
from curl_cffi import requests
from selectolax.parser import HTMLParser
from database_manager import CyclingDatabase


def clean_node_text(node):
    if not node:
        return ""
    try:
        text = node.text(separator=" ", strip=True)
    except TypeError:
        text = node.text()
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_full_page_text(tree):
    body = tree.css_first("body")
    if body:
        return clean_node_text(body)
    root = tree.root
    if root:
        return clean_node_text(root)
    return ""


def parse_birthdate_from_text(text):
    if not text:
        return None
    months_map = {
        "january": "01", "jan": "01", "february": "02", "feb": "02",
        "march": "03", "mar": "03", "april": "04", "apr": "04", "may": "05",
        "june": "06", "jun": "06", "july": "07", "jul": "07", "august": "08",
        "aug": "08", "september": "09", "sep": "09", "sept": "09",
        "october": "10", "oct": "10", "november": "11", "nov": "11",
        "december": "12", "dec": "12",
    }
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    months_regex = "|".join(months_map.keys())

    pattern_with_label = (
        rf"date\s+of\s+birth\s*:?\s*"
        rf"(\d{{1,2}})\s*(?:st|nd|rd|th)?\s*"
        rf"({months_regex})\s*,?\s*"
        rf"(\d{{4}})"
    )
    match = re.search(pattern_with_label, text, re.IGNORECASE)

    if not match:
        pattern_fallback = (
            rf"\b(\d{{1,2}})\s*(?:st|nd|rd|th)?\s*"
            rf"({months_regex})\s*,?\s*"
            rf"(\d{{4}})\b"
        )
        match = re.search(pattern_fallback, text, re.IGNORECASE)

    if not match:
        return None

    day = match.group(1).zfill(2)
    month_word = match.group(2).lower()
    year = match.group(3)
    month = months_map.get(month_word)
    if not month:
        return None
    return f"{year}-{month}-{day}"


def extract_birthdate(tree):
    page_text = get_full_page_text(tree)
    birthdate = parse_birthdate_from_text(page_text)
    debug_text = ""
    match_debug = re.search(r"date\s+of\s+birth.{0,120}", page_text, re.IGNORECASE)
    if match_debug:
        debug_text = match_debug.group(0)
    return birthdate, debug_text


def extract_nationality(tree):
    nationality = "UNKNOWN"
    nat_link = tree.css_first("a[href^='nation/']") or tree.css_first("a[href^='/nation/']")
    if nat_link:
        nat_text = clean_node_text(nat_link)
        if nat_text:
            nationality = nat_text.strip()[:3].upper()
    return nationality


def extract_weight_height_bmi(tree):
    page_text = get_full_page_text(tree)
    weight, height, bmi = None, None, None
    weight_match = re.search(r"Weight:\s*(\d+(?:\.\d+)?)", page_text, re.IGNORECASE)
    if weight_match:
        weight = float(weight_match.group(1))
    height_match = re.search(r"Height:\s*(\d+(?:\.\d+)?)", page_text, re.IGNORECASE)
    if height_match:
        height = float(height_match.group(1))
    if height and weight and height > 0:
        bmi = round(weight / (height ** 2), 2)
    return weight, height, bmi


def extract_current_team_and_tier(tree, target_lag_year):
    """
    Extrahiert Team und übersetzt das Team-Tier anhand der wissenschaftlichen Logik:
    WT / PT -> elite
    PRT / PCT / CT -> continental
    CLUB -> Überspringen (wird in Hauptschleife abgefangen)
    """
    current_team = "None / No Team"
    team_tier = "UNKNOWN"

    teams_ul = tree.css_first("ul.rdr-teams2")
    if not teams_ul:
        return current_team, team_tier

    for li in teams_ul.css("li"):
        season_div = li.css_first("div.season")
        name_div = li.css_first("div.name") or li.css_first("div.name2")

        if not season_div or not name_div:
            continue

        season_text = clean_node_text(season_div)
        if str(target_lag_year) not in season_text:
            continue

        team_link = name_div.css_first("a")
        if team_link:
            current_team = clean_node_text(team_link)

        name_text = clean_node_text(name_div)
        tier_match = re.search(r"\((WT|PT|PRT|PCT|CT|CLUB)\)", name_text, re.IGNORECASE)

        if tier_match:
            raw_tier = tier_match.group(1).upper()

            # Mapping-Logik anwenden
            if raw_tier in ["WT", "PT"]:
                team_tier = "elite"
            elif raw_tier in ["PRT", "PCT", "CT"]:
                team_tier = "continental"
            elif raw_tier == "CLUB":
                team_tier = "SKIP_CLUB"  # Signalwort für die Hauptschleife
        break

    return current_team, team_tier


def extract_lag_points_and_rank(tree, target_lag_year):
    lag_rider_points_season = 0.0
    lag_rider_rank_season = 9999

    ranking_table = tree.css_first("table.basic.small")
    if not ranking_table:
        return lag_rider_points_season, lag_rider_rank_season

    for tr in ranking_table.css("tbody tr"):
        tds = tr.css("td")
        if len(tds) < 3:
            continue

        year_link = tds[0].css_first("a")
        if not year_link:
            continue

        year_text = clean_node_text(year_link)
        if year_text != str(target_lag_year):
            continue

        points_div = tds[1].css_first("div.title")
        if points_div:
            points_text = clean_node_text(points_div).replace(",", "")
            try:
                lag_rider_points_season = float(points_text)
            except ValueError:
                lag_rider_points_season = 0.0

        rank_text = clean_node_text(tds[2]).replace("#", "").strip()
        if rank_text.isdigit():
            lag_rider_rank_season = int(rank_text)
        break

    return lag_rider_points_season, lag_rider_rank_season


def scrape_and_save_missing_profiles(df_missing_master, df_stage_meta):
    if df_missing_master.empty:
        print("Keine komplett fehlenden Fahrer-Profile. Schritt _04_ wird übersprungen.")
        return

    date_str = str(df_stage_meta["date"].iloc[0])
    try:
        stage_year = int(date_str.split()[-1])
    except Exception:
        stage_year = 2026
    target_lag_year = stage_year - 1

    print("\n==================================================================")
    print(f"DEEP-SCAN FEHLENDER PROFILE & LAGS ({target_lag_year}) (_04_")
    print(f"--- {len(df_missing_master)} Profile werden vollständig erfasst.")
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
        for _, row in df_missing_master.iterrows():
            url_slug = row["rider_url"]
            display_name = row["rider_name"]

            # Extraction der Team-Zugehörigkeit vorab für CLUB-Check
            full_url = f"https://www.procyclingstats.com/rider/{url_slug}"
            print(f"\nScrape Vollprofil via Selectolax: {display_name}")

            try:
                resp = requests.get(full_url, headers=headers, impersonate="chrome120", timeout=12)
            except Exception as request_error:
                print(f"Request-Fehler für {url_slug}: {request_error}")
                continue

            if resp.status_code != 200:
                print(f"Fehler: Konnte Seite für {url_slug} nicht laden. Status: {resp.status_code}")
                continue

            tree = HTMLParser(resp.text)

            # ==================================================================
            # CLUB-KICK-LOGIK: Prüfen, ob der Fahrer ignoriert werden muss
            # ==================================================================
            current_team, team_tier = extract_current_team_and_tier(tree, target_lag_year)

            if team_tier == "SKIP_CLUB":
                print(f"!!!!Fahrer {display_name} wird ignoriert (Tier: CLUB). Kein DB-Eintrag.")
                continue

            # ==================================================================
            # STAMMDATEN PARSEN
            # ==================================================================
            nationality = extract_nationality(tree)
            weight, height, bmi = extract_weight_height_bmi(tree)
            birthdate_str, birthdate_debug_text = extract_birthdate(tree)

            lag_rider_points_season, lag_rider_rank_season = extract_lag_points_and_rank(tree, target_lag_year)

            # ==================================================================
            # SCHREIBEN IN DIE DATENBANK
            # ==================================================================
            cursor.execute(
                """
                INSERT INTO r_master (
                    rider_url, meta_name, meta_url_name, height, weight, rider_bmi, nationality, birthdate
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rider_url) DO UPDATE SET
                    meta_name = COALESCE(excluded.meta_name, r_master.meta_name),
                    meta_url_name = COALESCE(excluded.meta_url_name, r_master.meta_url_name),
                    height = COALESCE(excluded.height, r_master.height),
                    weight = COALESCE(excluded.weight, r_master.weight),
                    rider_bmi = COALESCE(excluded.rider_bmi, r_master.rider_bmi),
                    nationality = COALESCE(excluded.nationality, r_master.nationality),
                    birthdate = COALESCE(excluded.birthdate, r_master.birthdate)
                """,
                (url_slug, display_name, url_slug, height, weight, bmi, nationality, birthdate_str),
            )

            # lags_historical: HIER ERZWINGEN WIR DAS UPDATE DER LEISTUNGSDATEN!
            # Kein COALESCE! Wenn die Primärschlüssel (rider_url + season_year) matchen,
            # werden die Punkte, Ränge und Teams mit den brandneuen PCS-Werten überschrieben.
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

            print(
                f"DB-Eintrag komplett! "
                f"Geburtstag: {birthdate_str} | "
                f"Tier: {team_tier} | "
                f"Punkte {target_lag_year}: {lag_rider_points_season} | "
                f"Rang: {lag_rider_rank_season}"
            )

        conn.commit()
        print("\n==================================================================")
        print("FEHLENDE PROFILE INKLUSIVE IHRER LAGS ERFOLGREICH GESPEICHERT!")
        print("==================================================================")

    except Exception as e:
        print(f"Fehler bei der Erfassung der Profildaten: {e}")
        conn.rollback()
    finally:
        conn.close()
