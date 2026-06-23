# Pipeline.py
from _01_input_handler import get_user_inputs
from _02_stage_scraper import scrape_pcs_stage_clean
from _03_database_checker import check_riders_against_db

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

    # Output DFs Kontrolle
    print("\n--- KONTROLLE METADATEN ---")
    print(df_stage_meta)

    print("\n--- KONTROLLE STARTLISTE (Kopfdaten) ---")
    print(df_startlist.head(5))

if __name__ == "__main__":
    run_pipeline()
