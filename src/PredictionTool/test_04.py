# run_wipe_and_test.py
import pandas as pd
from database_manager import CyclingDatabase
from _04_scraper_missing_drivers_profiles import scrape_and_save_missing_profiles

def wipe_and_test():
    db = CyclingDatabase(db_path="../../data/databases/cycling_production.db")
    conn = db.get_connection()
    cursor = conn.cursor()

    print("==================================================================")
    print(" 🧹 WIPE DATA: Lösche Filippo Magli aus allen Tabellen...")
    print("==================================================================")

    # Aus beiden Tabellen restlos entfernen
    cursor.execute("DELETE FROM r_master WHERE rider_url = 'filippo-magli2'")
    cursor.execute("DELETE FROM lags_historical WHERE rider_url = 'filippo-magli2'")
    conn.commit()

    # Kurzer Check, ob er wirklich weg ist
    cursor.execute("SELECT COUNT(*) FROM r_master WHERE rider_url = 'filippo-magli2'")
    print(f"➔ Verbleibende Einträge in r_master: {cursor.fetchone()[0]}")
    conn.close()

    print("\n==================================================================")
    print(" 🚀 STARTE TESTLAUF FÜR NEUANLAGE")
    print("==================================================================")

    # Test-DataFrames vorbereiten
    df_missing_test = pd.DataFrame([{
        "rider_name": "Magli Filippo",
        "rider_url": "filippo-magli2"
    }])
    df_meta_test = pd.DataFrame([{
        "date": "09 May 2026"
    }])

    # Jetzt den echten Scraper aufrufen
    scrape_and_save_missing_profiles(df_missing_test, df_meta_test)

    # ==================================================================
    # FINALE ERWEITERTE KONTROLLE (Alle Spalten & 2 Fahrer davor)
    # ==================================================================
    conn = db.get_connection()
    cursor = conn.cursor()

    # Pandas-Konfiguration für eine lückenlose Tabellendarstellung im Terminal
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # 1. Interne rowid von Filippo Magli ermitteln
    cursor.execute("SELECT rowid FROM r_master WHERE rider_url = 'filippo-magli2'")
    result_row = cursor.fetchone()

    if result_row:
        magli_rowid = result_row[0]

        print("\n------------------------------------------------------------------")
        print(" 📊 KONTROLLE NACH NEUANLAGE: r_master (Filippo + 2 davor)")
        print("------------------------------------------------------------------")
        # Holt alle Spalten explizit für das Dreiergespann (rowid - 2 bis rowid)
        cursor.execute('''
            SELECT rowid, rider_url, meta_name, meta_url_name, height, weight, rider_bmi, nationality, birthdate
            FROM r_master
            WHERE rowid BETWEEN ? AND ?
        ''', (magli_rowid - 2, magli_rowid))

        columns_master = ['rowid', 'rider_url', 'meta_name', 'meta_url_name', 'height', 'weight', 'rider_bmi', 'nationality', 'birthdate']
        print(pd.DataFrame(cursor.fetchall(), columns=columns_master).to_string(index=False))
    else:
        print("\n⚠️ Fehler: Filippo Magli wurde nach dem Scrapen nicht in r_master gefunden!")

    print("\n------------------------------------------------------------------")
    print(" 📊 KONTROLLE NACH NEUANLAGE: lags_historical (Filippo + 2 davor)")
    print("------------------------------------------------------------------")
    # Da lags_historical keine fortlaufenden rowids synchron zu r_master hat,
    # holen wir uns die Daten der letzten 3 Einträge, die hinzugefügt wurden
    cursor.execute('''
        SELECT rider_url, season_year, current_team, team_tier, lag_rider_points_season, lag_rider_rank_season
        FROM lags_historical
        ORDER BY rowid DESC
        LIMIT 3
    ''')

    # Die Liste umdrehen, damit Filippo (der neuste Eintrag) wie in Tabelle 1 unten steht
    rows_lags = cursor.fetchall()[::-1]
    columns_lags = ['rider_url', 'season_year', 'current_team', 'team_tier', 'lag_points', 'lag_rank']
    print(pd.DataFrame(rows_lags, columns=columns_lags).to_string(index=False))

    conn.close()

if __name__ == "__main__":
    wipe_and_test()
