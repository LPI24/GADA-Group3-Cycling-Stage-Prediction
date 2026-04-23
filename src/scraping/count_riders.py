import json

filename = 'riders_full_biometrics.jsonl'
total_entries = 0
unique_riders = set()

with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            name = data.get('url_name')
            if name:
                unique_riders.add(name)
            total_entries += 1
        except Exception:
            continue

print(f"--- Datensatz-Analyse ---")
print(f"Gesamtanzahl Zeilen: {total_entries}")
print(f"Eindeutige Fahrer-Profile: {len(unique_riders)}")
print(f"Dubletten gefunden: {total_entries - len(unique_riders)}")
