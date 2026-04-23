import re

# Einstellungen
LOG_FILE = "scraping_process_pc.log"
OUTPUT_FILE = "missing_riders_analysis.txt"
# Zeitraum von 11:42 bis zum Ende
START_TIME = "2026-04-23 11:42"

missing_riders = []
error_pattern = re.compile(r"❌ Schwerer Fehler bei (.*?):")

with open(LOG_FILE, "r", encoding="utf-8") as f:
    for line in f:
        # Prüfen, ob die Zeile zeitlich relevant ist
        if line >= START_TIME:
            match = error_pattern.search(line)
            if match:
                rider_name = match.group(1)
                missing_riders.append(rider_name)

# Ergebnis speichern
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for rider in missing_riders:
        f.write(f"{rider}\n")

print(f"--- Analyse beendet ---")
print(f"Zeitraum ab: {START_TIME}")
print(f"Fehlerhafte Fahrer gefunden: {len(missing_riders)}")
print(f"Liste gespeichert in: {OUTPUT_FILE}")
