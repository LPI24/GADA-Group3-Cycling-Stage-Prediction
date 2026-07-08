# 04_scraper_missing_drivers_profiles.py
#
# Dieses Skript ergänzt fehlende Fahrerprofile und historische Lag-Daten
# durch Scraping der ProCyclingStats-Fahrerseiten. Es extrahiert Stamm-
# daten (Geburtstag, Nationalität, Größe, Gewicht, BMI) sowie Lag-Daten
# (Punkte, Rang, Team) für das Vorjahr und schreibt diese in die SQLite-
# Datenbank.

import re
from datetime import datetime
import pandas as pd
from curl_cffi import requests
from selectolax.parser import HTMLParser
from database_manager import CyclingDatabase


# ---------------------------------------------------------------------------
# HILFSFUNKTIONEN ZUR TEXTEXTRAKTION
# ---------------------------------------------------------------------------

def clean_node_text(node):
    """Extrahiert und bereinigt den Textinhalt eines HTML-Knotens.

    Ersetzt geschützte Leerzeichen (\xa0) und kollabiert mehrfache
    Leerzeichen zu einem einzigen.
    """
    if not node:
        return ""
    try:
        # selectolax: text(separator=" ") fügt Leerzeichen zwischen
        # Kind-Elementen ein, statt sie direkt zusammenzufügen
        text = node.text(separator=" ", strip=True)
    except TypeError:
        # Fallback, falls der Knoten kein separator-Argument unterstützt
        text = node.text()
    text = text.replace("\xa0", " ")  # non-breaking space -> normale Leerzeichen
    text = re.sub(r"\s+", " ", text)  # mehrfache Leerzeichen -> eins
    return text.strip()


def get_full_page_text(tree):
    """Gibt den gesamten sichtbaren Text der HTML-Seite zurück.
    Wird genutzt, um per Regex strukturierte Werte (z. B. Geburtsdatum,
    Gewicht, Größe) aus dem Fließtext zu extrahieren.
    """
    body = tree.css_first("body")
    if body:
        return clean_node_text(body)
    root = tree.root
    if root:
        return clean_node_text(root)
    return ""


# ---------------------------------------------------------------------------
# EXTRAKTION: GEBURTSDATUM
# ---------------------------------------------------------------------------

def parse_birthdate_from_text(text):
    """Parst ein Geburtsdatum aus Fließtext.

    Versucht zwei Regex-Muster:
      1. Mit Label "Date of birth:" (präziser)
      2. Fallback: beliebiges "DD Month YYYY" im Text (unspezifischer)

    Gibt das Datum im Format YYYY-MM-DD zurück oder None bei Misserfolg.
    """
    if not text:
        return None

    # Mapping von Monatsnamen (voll und abgekürzt) auf Monatsnummern
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

    # Muster 1: "Date of birth: 15 March 1990"
    pattern_with_label = (
        rf"date\s+of\s+birth\s*:?\s*"
        rf"(\d{{1,2}})\s*(?:st|nd|rd|th)?\s*"          # Tag mit optionalem Ordinal
        rf"({months_regex})\s*,?\s*"                    # Monatsname
        rf"(\d{{4}})"                                    # Jahr
    )
    match = re.search(pattern_with_label, text, re.IGNORECASE)

    # Muster 2 (Fallback): "15 March 1990" ohne Label
    if not match:
        pattern_fallback = (
            rf"\b(\d{{1,2}})\s*(?:st|nd|rd|th)?\s*"
            rf"({months_regex})\s*,?\s*"
            rf"(\d{{4}})\b"
        )
        match = re.search(pattern_fallback, text, re.IGNORECASE)

    if not match:
        return None

    # Gefundene Gruppen in ISO-Format YYYY-MM-DD umwandeln
    day = match.group(1).zfill(2)
    month_word = match.group(2).lower()
    year = match.group(3)
    month = months_map.get(month_word)
    if not month:
        return None
    return f"{year}-{month}-{day}"


def extract_birthdate(tree):
    """Kombiniert Seitentext-Extraktion und Geburtsdatum-Parsing.
    Gibt zusätzlich einen Debug-String zurück, der den Textbereich um
    "Date of birth" enthält – nützlich bei Fehlersuche.
    """
    page_text = get_full_page_text(tree)
    birthdate = parse_birthdate_from_text(page_text)
    debug_text = ""
    match_debug = re.search(r"date\s+of\s+birth.{0,120}", page_text, re.IGNORECASE)
    if match_debug:
        debug_text = match_debug.group(0)
    return birthdate, debug_text


# ---------------------------------------------------------------------------
# EXTRAKTION: NATIONALITÄT
# ---------------------------------------------------------------------------

def extract_nationality(tree):
    """Extrahiert die Nationalität eines Fahrers anhand des Links auf
    die Nationenseite (z. B. /nation/belgium). Gibt einen 3-Buchstaben-
    Code (ISO-ähnlich) in Großbuchstaben zurück, z. B. 'BEL'.
    Falls kein Nationenlink gefunden wird, ist die Nationalität 'UNKNOWN'.
    """
    nationality = "UNKNOWN"
    nat_link = tree.css_first("a[href^='nation/']") or tree.css_first("a[href^='/nation/']")
    if nat_link:
        nat_text = clean_node_text(nat_link)
        if nat_text:
            nationality = nat_text.strip()[:3].upper()
    return nationality


# ---------------------------------------------------------------------------
# EXTRAKTION: KÖRPERMASSE (GEWICHT, GRÖßE, BMI)
# ---------------------------------------------------------------------------

def extract_weight_height_bmi(tree):
    """Extrahiert Gewicht (kg) und Größe (m) aus dem Seitentext und
    berechnet daraus den BMI = Gewicht / Größe².
    Gibt (weight, height, bmi) zurück; nicht gefundene Werte sind None.
    """
    page_text = get_full_page_text(tree)
    weight, height, bmi = None, None, None

    weight_match = re.search(r"Weight:\s*(\d+(?:\.\d+)?)", page_text, re.IGNORECASE)
    if weight_match:
        weight = float(weight_match.group(1))

    height_match = re.search(r"Height:\s*(\d+(?:\.\d+)?)", page_text, re.IGNORECASE)
    if height_match:
        height = float(height_match.group(1))

    # BMI nur berechnen, wenn beide Werte vorhanden und Größe > 0
    if height and weight and height > 0:
        bmi = round(weight / (height ** 2), 2)

    return weight, height, bmi


# ---------------------------------------------------------------------------
# EXTRAKTION: TEAM UND TEAM-TIER
# ---------------------------------------------------------------------------

def extract_current_team_and_tier(tree, target_lag_year):
    """
    Extrahiert das Team eines Fahrers für das Ziel-Lag-Jahr und übersetzt
    das PCS-Team-Tier in ein vereinfachtes Schema:

        WT / PT     -> elite
        PRT / PCT / CT -> continental
        CLUB        -> SKIP_CLUB (Signalwort: Fahrer wird in der
                      Hauptschleife übersprungen, da keine relevante
                      Profilierung auf Pro-Tour-Niveau)

    Rückgabe: (current_team_name, team_tier)
    """
    current_team = "None / No Team"
    team_tier = "UNKNOWN"

    # PCS listet Teams in einer <ul class="rdr-teams2"> mit <li>-Einträgen
    teams_ul = tree.css_first("ul.rdr-teams2")
    if not teams_ul:
        return current_team, team_tier

    for li in teams_ul.css("li"):
        season_div = li.css_first("div.season")
        name_div = li.css_first("div.name") or li.css_first("div.name2")

        if not season_div or not name_div:
            continue

        # Nur den Eintrag berücksichtigen, der zum Ziel-Lag-Jahr passt
        season_text = clean_node_text(season_div)
        if str(target_lag_year) not in season_text:
            continue

        # Teamnamen aus dem Link extrahieren
        team_link = name_div.css_first("a")
        if team_link:
            current_team = clean_node_text(team_link)

        # Tier-Kürzel in Klammern suchen, z. B. "(WT)" oder "(PCT)"
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


# ---------------------------------------------------------------------------
# EXTRAKTION: LAG-PUNKTE UND LAG-RANG
# ---------------------------------------------------------------------------

def extract_lag_points_and_rank(tree, target_lag_year):
    """Extrahiert die PCS-Punkte und den PCS-Rang des Fahrers für das
    Ziel-Lag-Jahr aus der Ranking-Tabelle.

    Gibt (lag_points, lag_rank) zurück. Standardwerte sind 0.0 Punkte
    und Rang 9999, falls keine Daten gefunden werden.
    """
    lag_rider_points_season = 0.0
    lag_rider_rank_season = 9999

    # PCS Ranking-Tabelle: <table class="basic small">
    ranking_table = tree.css_first("table.basic.small")
    if not ranking_table:
        return lag_rider_points_season, lag_rider_rank_season

    for tr in ranking_table.css("tbody tr"):
        tds = tr.css("td")
        if len(tds) < 3:
            continue

        # Erste Spalte: Jahr-Link
        year_link = tds[0].css_first("a")
        if not year_link:
            continue

        year_text = clean_node_text(year_link)
        if year_text != str(target_lag_year):
            continue

        # Zweite Spalte: Punkte (in einem div.title verschachtelt)
        points_div = tds[1].css_first("div.title")
        if points_div:
            points_text = clean_node_text(points_div).replace(",", "")
            try:
                lag_rider_points_season = float(points_text)
            except ValueError:
                lag_rider_points_season = 0.0

        # Dritte Spalte: Rang (mit führendem #, z. B. "#42")
        rank_text = clean_node_text(tds[2]).replace("#", "").strip()
        if rank_text.isdigit():
            lag_rider_rank_season = int(rank_text)
        break

    return lag_rider_points_season, lag_rider_rank_season


# ---------------------------------------------------------------------------
# HAUPTFUNKTION: FEHLENDE PROFILE SCRAPEN UND SPEICHERN
# ---------------------------------------------------------------------------

def scrape_and_save_missing_profiles(df_missing_master, df_stage_meta):
    """Hauptfunktion: Scrapt vollständige Profile für alle in
    df_missing_master gelisteten Fahrer von ProCyclingStats und schreibt
    die extrahierten Stamm- und Lag-Daten in die SQLite-Datenbank.

    Parameter:
        df_missing_master: DataFrame mit Spalten 'rider_url' und 'rider_name'
                           für alle Fahrer, deren Profile noch fehlen.
        df_stage_meta:     DataFrame mit Etappenmetadaten (mindestens
                           Spalte 'date'), um das Ziel-Lag-Jahr abzuleiten.
    """

    # Abbruch, wenn keine fehlenden Profile vorhanden sind
    if df_missing_master.empty:
        print("Keine komplett fehlenden Fahrer-Profile. Schritt 04 wird übersprungen.")
        return

    # Ziel-Lag-Jahr aus dem Etappendatum ableiten (Etappenjahr - 1)
    date_str = str(df_stage_meta["date"].iloc[0])
    try:
        stage_year = int(date_str.split()[-1])
    except Exception:
        stage_year = 2026
    target_lag_year = stage_year - 1

    print("\n==================================================================")
    print(f"DEEP-SCAN FEHLENDER PROFILE & LAGS ({target_lag_year}) (04)")
    print(f"--- {len(df_missing_master)} Profile werden vollständig erfasst.")
    print("==================================================================")

    # Datenbankverbindung aufbauen
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
            url_slug = row["rider_url"]      # PCS-URL-slug z. B. "tadej-pogacar"
            display_name = row["rider_name"]

            # PCS-Fahrerseite laden
            full_url = f"https://www.procyclingstats.com/rider/{url_slug}"
            print(f"\nScrape Vollprofil via Selectolax: {display_name}")

            try:
                # curl_cffi mit Chrome-Impersonierung, um Anti-Bot-Schutz zu umgehen
                resp = requests.get(full_url, headers=headers, impersonate="chrome120", timeout=12)
            except Exception as request_error:
                print(f"Request-Fehler für {url_slug}: {request_error}")
                continue

            if resp.status_code != 200:
                print(f"Fehler: Konnte Seite für {url_slug} nicht laden. Status: {resp.status_code}")
                continue

            # HTML parsen
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

            # Lag-Daten (Punkte + Rang für das Vorjahr) extrahieren
            lag_rider_points_season, lag_rider_rank_season = extract_lag_points_and_rank(tree, target_lag_year)

            # ==================================================================
            # SCHREIBEN IN DIE DATENBANK: Tabelle r_master (Stammdaten)
            # ==================================================================
            # INSERT ... ON CONFLICT: Falls der Fahrer bereits existiert,
            # werden nur die Felder aktualisiert, die COALESCE als neuen
            # Wert zulässt (also nur, wenn der neue Wert nicht NULL ist).
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

            # ==================================================================
            # SCHREIBEN IN DIE DATENBANK: Tabelle lags_historical (Leistungsdaten)
            # ==================================================================
            # HIER ERZWINGEN WIR DAS UPDATE DER LEISTUNGSDATEN!
            # Kein COALESCE! Wenn die Primärschlüssel (rider_url + season_year)
            # matchen, werden die Punkte, Ränge und Teams mit den brandneuen
            # PCS-Werten überschrieben.
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

        # Alle Änderungen committen, nachdem jeder Fahrer verarbeitet wurde
        conn.commit()
        print("\n==================================================================")
        print("FEHLENDE PROFILE INKLUSIVE IHRER LAGS ERFOLGREICH GESPEICHERT!")
        print("==================================================================")

    except Exception as e:
        # Bei einem Fehler alle Änderigkeiten zurückrollen (Transaktionssicherheit)
        print(f"Fehler bei der Erfassung der Profildaten: {e}")
        conn.rollback()
    finally:
        # Verbindung in jedem Fall schließen
        conn.close()
