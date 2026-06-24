# Pipeline.py
from _01_input_handler import get_user_inputs
from _02_stage_scraper import scrape_pcs_stage_clean
from _03_database_checker import check_riders_against_db
from _04_scraper_missing_drivers_profiles import scrape_and_save_missing_profiles
from _05_scraper_missing_lags import scrape_and_save_missing_lags
from database_manager import CyclingDatabase
import pandas as pd

def run_pipeline():
    # 1. Inputs vom User abfragen (Giro, Tour, Vuelta)
    user_inputs = get_user_inputs()

    # 2. Den neuen Scraper füttern
    df_startlist, df_stage_meta = scrape_pcs_stage_clean(user_inputs["pcs_url"], user_inputs)

    # Sicherheits-Stopp, falls beim Scraping was schiefging
    if df_startlist.empty or df_stage_meta.empty:
        print("Abbruch: Scraping fehlgeschlagen.")
        return

    # 3. Datenbank-Check ausführen
    df_missing_master, df_missing_lags = check_riders_against_db(df_startlist, df_stage_meta)

    # 4. Unbekannte Fahrerprofile UND deren Lags live in einem Abwasch scrapen!
    scrape_and_save_missing_profiles(df_missing_master, df_stage_meta)

    # Schritt 5: rstlichen Fahrer lag_daten updaten
    scrape_and_save_missing_lags(df_missing_lags, df_stage_meta)

    print("\n--- Pipeline-Durchlauf beendet ---")
    print(50*"=")

    # ==================================================================
    # FINALE KONSOLIDIERTE STARTLISTE AUS DER DB
    # ==================================================================
    print("\n" + "="*80)
    print(" FINALE KONSOLIDIERTE STARTLISTE (ALLE DATEN AUS DER DATENBANK)")
    print("="*80)

    # Ziel-Vorjahr für die Lag-Filterung berechnen
    date_str = str(df_stage_meta["date"].iloc[0])
    try:
        stage_year = int(date_str.split()[-1])
    except Exception:
        stage_year = 2026
    target_lag_year = stage_year - 1

    # Verbindung zur DB aufbauen
    db = CyclingDatabase(db_path="../../data/databases/cycling_production.db")
    conn = db.get_connection()

    # Liste aller rider_urls aus der aktuellen Startliste für die SQL-Query aufbereiten
    startlist_urls = df_startlist['rider_url'].tolist()

    if startlist_urls:
        # Pandas-Optionen für eine perfekte, ungeschnittene Terminal-Tabelle
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)  # Zeigt alle 184 Fahrer komplett an
        pd.set_option('display.width', 1200)

        # SQL Query mit einem INNER JOIN über beide Tabellen
        # Filtert stur nach den URLs der Startliste und dem passenden Vorjahr!
        query = f"""
            SELECT
                m.rider_url,
                m.meta_name AS name,
                m.nationality AS nat,
                m.birthdate,
                m.height,
                m.weight,
                m.rider_bmi AS bmi,
                l.season_year AS lag_year,
                l.current_team AS team,
                l.team_tier AS tier,
                l.lag_rider_points_season AS points,
                l.lag_rider_rank_season AS rank
            FROM r_master m
            INNER JOIN lags_historical l ON m.rider_url = l.rider_url
            WHERE m.rider_url IN ({','.join(['?']*len(startlist_urls))})
              AND l.season_year = ?
        """

        # Parameter übergeben: Erst alle URLs, am Ende das target_lag_year
        params = startlist_urls + [target_lag_year]

        df_final_report = pd.read_sql_query(query, conn, params=params)

        # Ausgabe der Master-Tabelle
        if not df_final_report.empty:
            print(df_final_report.to_string(index=False))
            print("\n" + "-"*80)
            print(f"Gesamt-Statistik: {len(df_final_report)} von {len(df_startlist)} Fahrern erfolgreich mit Vorjahres-Lags geladen.")
            print("-"*80)
        else:
            print("--Warnung: Keine passenden Einträge für die Startliste in der Datenbank gefunden.")
    else:
        print("--Fehler: Startliste ist leer.")

    conn.close()

if __name__ == "__main__":
    run_pipeline()
