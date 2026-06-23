# _03_database_checker.py
import pandas as pd
from database_manager import CyclingDatabase

def check_riders_against_db(df_startlist, df_stage_meta):
    """
    Gleicht die gescrapte Startliste mit der SQLite-Datenbank ab.

    1. Prüft, ob Fahrer komplett in r_master (Tabelle 1) fehlen.
    2. Prüft, ob für die vorhandenen Fahrer die Lags für die aktuelle Saison in lags_historical (Tabelle 2) fehlen.
    """
    print("\n==================================================================")
    print("STARTE DATENBANK-INTEGRITÄTS-CHECK (_03_)")
    print("==================================================================")

    # Instanziiere den DB-Manager
    db = CyclingDatabase(db_path="../../data/databases/cycling_production.db")

    # Saison-Jahr aus den Etappen-Metadaten ziehen
    current_season = int(df_stage_meta['stage_nr'].iloc[0]) # Fallback/Zuweisung je nach Struktur, hier nutzen wir das Jahr aus der Pipeline
    # Da wir das Jahr dynamisch brauchen, holen wir es sicherheitshalber aus dem Datum oder übergeben es direkt.
    # Für den Abgleich extrahieren wir das Jahr aus dem user_input/Metadaten:
    # Hinweis: Da df_stage_meta das Datum hat (z.B. "09 May 2026"), extrahieren wir die Jahreszahl:
    date_str = str(df_stage_meta['date'].iloc[0])
    try:
        stage_year = int(date_str.split()[-1]) # Extrahiert "2026" aus "09 May 2026"
    except:
        stage_year = 2026 # Sicherer Fallback

    print(f"➔ Prüfe historische Lags für das Ziel-Saisonjahr: {stage_year}")

    missing_master = []  # Liste 1: Fahrer fehlen komplett in Tabelle 1
    missing_lags = []    # Liste 2: Fahrer sind in Tabelle 1, aber Lags für dieses Jahr fehlen in Tabelle 2

    # Loop durch alle Fahrer der gescrapten Startliste
    for _, row in df_startlist.iterrows():
        url = row['rider_url']
        name = row['rider_name']

        # CHK 1: Stammdaten prüfen
        df_master = db.get_rider_master(url)

        if df_master is None:
            # Fahrer fehlt komplett in der Master-Tabelle
            missing_master.append({"rider_url": url, "rider_name": name})
        else:
            # CHK 2: Wenn er in Master existiert, prüfen wir die beweglichen Lags für genau DIESES Etappenjahr
            df_lags = db.get_rider_lags(url, stage_year)
            if df_lags is None:
                # Fahrer hat Stammdaten, aber keine Lags für diese Saison
                missing_lags.append({"rider_url": url, "rider_name": name})

    # In DataFrames umwandeln für die Weiterverarbeitung
    df_missing_master = pd.DataFrame(missing_master)
    df_missing_lags = pd.DataFrame(missing_lags)

  #Ausgabe der Ergebnisse
    print("\n------------------------------------------------------------------")
    print(f"ERGEBNIS 1: Komplett unbekannte Fahrer (fehlen in r_master): {len(df_missing_master)}")
    print("------------------------------------------------------------------")
    if not df_missing_master.empty:
        print(df_missing_master[['rider_name', 'rider_url']].to_string(index=False))
    else:
        print("➔ Sensationell! Alle Fahrer der Startliste besitzen biometrische Stammdaten.")

    print("\n------------------------------------------------------------------")
    print(f"ERGEBNIS 2: Vorhandene Fahrer, denen Saison-Lags ({stage_year}) fehlen: {len(df_missing_lags)}")
    print("------------------------------------------------------------------")
    if not df_missing_lags.empty:
        print(df_missing_lags[['rider_name', 'rider_url']].to_string(index=False))
    else:
        print(f"➔ Perfekt! Für alle qualifizierten Fahrer liegen die Leistungs-Lags für {stage_year} vor.")

    print("==================================================================\n")

    return df_missing_master, df_missing_lags
