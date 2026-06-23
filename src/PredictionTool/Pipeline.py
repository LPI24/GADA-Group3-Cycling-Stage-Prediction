from _01_input_handler import get_user_inputs
from _02_stage_scraper import scrape_pcs_stage_clean
def run_pipeline():
    # Inputs vom User abfragen (Giro, Tour, Vuelta)
    user_inputs = get_user_inputs()

    # Den neuen Scraper füttern
    df_startlist, df_stage_meta = scrape_pcs_stage_clean(user_inputs["pcs_url"], user_inputs)

    # Output DFs FahrerRangliste oder Starterliste und Metadaten der Etappe
    print("\n--- KONTROLLE METADATEN ---")
    print(df_stage_meta)

    print("\n--- KONTROLLE STARTLISTE (Kopfdaten) ---")
    print(df_startlist.head(5))

if __name__ == "__main__":
    run_pipeline()
