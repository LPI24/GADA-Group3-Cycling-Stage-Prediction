import json
import os

# Pfad-Logik
base_path = os.path.dirname(os.path.abspath(__file__))
BIOMETRICS_FILE = os.path.abspath(os.path.join(base_path, "../../../data/raw/riders_full_biometrics.jsonl"))
CLEAN_FILE = os.path.abspath(os.path.join(base_path, "../../../data/raw/riders_full_biometrics_clean.jsonl"))

def deduplicate_riders():
    if not os.path.exists(BIOMETRICS_FILE):
        print(f"❌ Datei nicht gefunden: {BIOMETRICS_FILE}")
        return

    riders_dict = {}
    total_count = 0
    duplicate_count = 0

    print("Lese Profile und entferne Duplikate...")

    with open(BIOMETRICS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rider = json.loads(line)
                url = rider.get("url_name")

                if url:
                    if url in riders_dict:
                        duplicate_count += 1
                    # Wir speichern den Rider im Dict.
                    # Wenn die URL nochmal kommt, wird sie einfach überschrieben (Update-Logik)
                    riders_dict[url] = line
                    total_count += 1
            except Exception as e:
                print(f"⚠️ Fehler in Zeile: {e}")

    # Speichere die bereinigten Daten
    with open(CLEAN_FILE, "w", encoding="utf-8") as f:
        for url in riders_dict:
            f.write(riders_dict[url] + "\n")

    print("\n" + "="*30)
    print("STATISTIK BEREINIGUNG")
    print("="*30)
    print(f"Zeilen insgesamt gelesen:    {total_count}")
    print(f"Duplikate entfernt:          {duplicate_count}")
    print(f"Einzigartige Profile übrig:  {len(riders_dict)}")
    print("="*30)
    print(f"Bereinigte Datei gespeichert unter: {CLEAN_FILE}")

if __name__ == "__main__":
    deduplicate_riders()
