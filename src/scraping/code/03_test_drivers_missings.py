import pandas as pd
import os

def main():
# 1. Erst den Pfad des Skripts holen
    base_path = os.path.dirname(os.path.abspath(__file__))

    # 2. Dann 3 Ebenen hoch (code -> scraping -> src -> cycling_project)
    project_root = os.path.normpath(os.path.join(base_path, "../../../"))

    # Pfade zu den Dateien
    results_path = os.path.join(project_root, "data", "raw", "V1_grand_tour_stage_results.jsonl")
    riders_path = os.path.join(project_root, "data", "raw", "riders_full_biometrics.jsonl")

    # Kleiner Test-Print für dich zum Checken
    print(f"Suche Datei in: {results_path}")

    try:
        # 1. Etappendaten laden und explodieren
        df_stages = pd.read_json(results_path, lines=True)
        df_results = df_stages.explode('results')
        #Extrahieren des Result Teils

        # 2. Den rider_url Key extrahieren
        # Bspw. ziehen 'david-zabriskie' aus dem Dictionary
        df_results['rider_id'] = df_results['results'].apply(
            lambda x: x.get('rider_url') if isinstance(x, dict) else None
        )
        #print(df_results['rider_id'])

        # 3. Fahrer-Biometrie laden
        df_riders = pd.read_json(riders_path, lines=True)
    except Exception as e:
        print(f"Fehler beim Laden: {e}")
        return

    # 4. Abgleich über die URL-Strings
    # In der Biometrie-Datei heißt das Feld 'url_name'
    stage_ids = set(df_results['rider_id'].dropna().unique())
    biometry_ids = set(df_riders['url_name'].dropna().unique())

    found = stage_ids.intersection(biometry_ids)
    missing = stage_ids - biometry_ids

    # Statistik
    total = len(stage_ids)
    coverage = (len(found) / total) * 100 if total > 0 else 0

    print("\n" + "="*30)
    print(f"ERGEBNIS DES ABGLEICHS (via URL-ID)")
    print("="*30)
    print(f"Einzigartige Fahrer-IDs in Rennen: {total}")
    print(f"Davon Profile gefunden:            {len(found)}")
    print(f"Fahrer ohne Profil:                {len(missing)}")
    print(f"Abdeckungsrate:                    {coverage:.2f}%")
    print("="*30)

    if missing:
        print("\nTOP FEHLENDE RIDER-URLs (Wichtig für Nach-Scraping):")
        # Häufigkeit der fehlenden URLs in den Rennergebnissen
        missing_list = df_results[df_results['rider_id'].isin(missing)]
        print(missing_list['rider_id'].value_counts().head(30))

if __name__ == "__main__":
    main()
