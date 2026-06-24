# check_scraped_riders.py
import pandas as pd
from database_manager import CyclingDatabase

def show_riders():
    db = CyclingDatabase(db_path="../../data/databases/cycling_production.db")
    conn = db.get_connection()

    # Zielfahrer für die Abfrage definieren
    target_urls = [

        'adam-yates',
        'tord-gudmestad'

    ]

    # Pandas-Konfiguration für schicke Terminal-Ausgabe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("==================================================================")
    print(" 📊 KONTROLLE: r_master")
    print("==================================================================")
    # SQL-Abfrage mit IN-Klausel für die r_master
    query_master = f"""
        SELECT rider_url, meta_name, meta_url_name, height, weight, rider_bmi, nationality, birthdate
        FROM r_master
        WHERE rider_url IN ({','.join(['?']*len(target_urls))})
    """
    df_master = pd.read_sql_query(query_master, conn, params=target_urls)
    print(df_master.to_string(index=False))

    print("\n==================================================================")
    print(" 📊 KONTROLLE: lags_historical")
    print("==================================================================")
    # SQL-Abfrage mit IN-Klausel für die lags_historical
    query_lags = f"""
        SELECT rider_url, season_year, current_team, team_tier, lag_rider_points_season, lag_rider_rank_season
        FROM lags_historical
        WHERE rider_url IN ({','.join(['?']*len(target_urls))})
    """
    df_lags = pd.read_sql_query(query_lags, conn, params=target_urls)
    print(df_lags.to_string(index=False))

    conn.close()

if __name__ == "__main__":
    show_riders()
