# test_db.py
import os
from database_manager import CyclingDatabase

def run_database_tests():
    # 1. Instanziiere den Manager
    # Hinweis: Da wir uns im selben Ordner befinden, prüfen wir den Pfad zur DB
    db = CyclingDatabase(db_path="../../data/databases/cycling_production.db")

    # --- TEST 1: Wie viele Fahrer haben wir insgesamt? ---
    print("\n[Test 1] Gesamtanzahl der Fahrer in der Datenbank:")
    df_count = db.query("SELECT COUNT(*) as total_riders FROM r_master")
    print(f"➔ Anzahl geladener Fahrer: {df_count['total_riders'].iloc[0]}")

    # --- TEST 2: Gezielte Abfrage über die eingebaute Methode ---
    print("\n[Test 2] Stammdaten von Tadej Pogačar abfragen:")
    # Wir nutzen die fertige Methode aus deinem Manager
    df_pogacar = db.get_rider_master("tadej-pogacar")

    if df_pogacar is not None:
        print(df_pogacar[['rider_url', 'meta_name', 'height', 'weight', 'birthdate']].to_string(index=False))
    else:
        print("Fahrer 'tadej-pogacar' nicht in der DB gefunden (evtl. URL prüfen).")

    # --- TEST 3: Ein komplexerer SQL JOIN ---
    print("\n[Test 3] Freie SQL-Abfrage: Top 5 Fahrer im Jahr 2024 (nach PCS-Punkten):")
    sql_join = """
        SELECT m.meta_name, l.season_year, l.current_team, l.lag_rider_points_season
        FROM r_master m
        JOIN lags_historical l ON m.rider_url = l.rider_url
        WHERE l.season_year = 2024
        ORDER BY l.lag_rider_points_season DESC
        LIMIT 5
    """
    df_top5 = db.query(sql_join)
    print(df_top5.to_string(index=False))

    # --- TEST 4: Verteilung der Team Tiers ---
    print("\n[Test 4] Freie SQL-Abfrage: Wie oft sind die Team-Tiers im Datensatz vertreten?")
    sql_group = """
        SELECT team_tier, COUNT(*) as anzahl
        FROM lags_historical
        GROUP BY team_tier
        ORDER BY anzahl DESC
    """
    df_tiers = db.query(sql_group)
    print(df_tiers.to_string(index=False))


    print("Tests durchlaufen")


if __name__ == "__main__":
    run_database_tests()
