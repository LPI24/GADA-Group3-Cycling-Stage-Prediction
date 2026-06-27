# Data Pipeline für die Modelle
- Ziel: einheitliche Datenstruktur für die anschließenden Modelle


```python
import pandas as pd
import numpy as np
import os
```


```python
pickle_path = '../../data/processed/26_cleaned_master_data.pkl'

df = pd.read_pickle(pickle_path)
```


```python
print(f"Ursprünglicher Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten.")
```

    Ursprünglicher Datensatz geladen: 196048 Zeilen, 46 Spalten.
    


```python
for idx, col in enumerate(df.columns, 1):
    print(f"{idx:02d}. {col} ({df[col].dtype})")

# Spaltennamen inkl Datentypen
```

    01. meta_race (category)
    02. meta_year (int64)
    03. meta_url (object)
    04. rank (float64)
    05. meta_rider_url (object)
    06. height (float64)
    07. meta_name (object)
    08. meta_nationality (category)
    09. weight (int64)
    10. meta_url_name (object)
    11. meta_departure (object)
    12. meta_arrival (object)
    13. distance (float64)
    14. vertical_meters (int64)
    15. one_day_races (int64)
    16. gc (int64)
    17. time_trial (int64)
    18. sprint (int64)
    19. climber (int64)
    20. hills (int64)
    21. stage_nr (int64)
    22. meta_date (datetime64[ns])
    23. meta_departure_lat (float64)
    24. meta_departure_lon (float64)
    25. meta_arrival_lat (float64)
    26. meta_arrival_lon (float64)
    27. meta_rider_points_season (int64)
    28. meta_rider_rank_season (int64)
    29. meta_current_team (object)
    30. team_tier (category)
    31. age_at_race (int64)
    32. rider_bmi (float64)
    33. meta_race_competitiveness_median (float64)
    34. meta_team_power_index (float64)
    35. wind_stability_index (float64)
    36. weather_temp_mean (float64)
    37. weather_temp_trend (float64)
    38. weather_rain_prob_mean (float64)
    39. weather_precipitation_mean (float64)
    40. weather_humidity_mean (float64)
    41. won_how_cat (category)
    42. gradient_final_km (float64)
    43. lag_rider_points_season (float64)
    44. lag_rider_rank_season (float64)
    45. lag_race_competitiveness_median (float64)
    46. lag_team_power_index (float64)
    

# Eindeutige Rennkombination (STAGE_ID) für den Ranker erzeugen
- Wir kombinieren das Rennen, das Jahr und die Etappennummer zu einer eindeutigen ID


```python

df['stage_id'] = df['meta_race'].astype(str) + "_" + df['meta_year'].astype(str) + "_ST" + df['stage_nr'].astype(str)
```

# Targets dynamisch generieren

Um dem inhärent ordinalen Charakter von Platzierungsschätzungen im Radsport gerecht zu werden, wurde von einer redundanten Modellierung separater, binärer Klassifikationsgrenzen (Top-5/Top-10/Top-20) abgesehen. Ein solches Vorgehen verwirft die ordinale Struktur der Zielvariable und führt zu einem erheblichen Informationsverlust, da beispielsweise knappe Platzierungen knapp außerhalb der Schwellenwerte mathematisch mit dem hinteren Hauptfeld gleichgesetzt werden. Stattdessen wird ein Ordinales Klassifikations-Framing über eine integrierte Relevanz-Skala (Y∈{0,1,2,3}) implementiert. Dies sichert die logische Konsistenz der Klassengrenzen, sensitiviert die Verlustfunktion für graduelle Fehlprognosen und minimiert den informationstheoretischen Verlust vor dem Übergang zum Listwise-Ranking.


```python
# Bedingungen für ordinale Struktur definieren (Strengste Bedingung zuerst!)
conditions = [
    (df['rank'] <= 5),
    (df['rank'] <= 10),
    (df['rank'] <= 20)
]

# Die hierarchischen Relevanz-Labels zuweisen
values = [3, 2, 1]

# Gemeinsame ordinale Zielvariable erstellen (Standardwert 0 = Hauptfeld)
df['target_ordinal_relevance'] = np.select(conditions, values, default=0)


print(df['target_ordinal_relevance'].value_counts().sort_index(ascending=False))
```

    target_ordinal_relevance
    3      5851
    2      5829
    1     11628
    0    172740
    Name: count, dtype: int64
    

# Aufteilen der Daten


```python
# Gruppe 1: Die Features (X)
features_benchmark = [
    'distance', 'vertical_meters', 'stage_nr', 'team_tier', 'age_at_race',
    'rider_bmi', 'wind_stability_index', 'weather_temp_mean', 'weather_temp_trend',
    'weather_rain_prob_mean', 'weather_precipitation_mean', 'weather_humidity_mean',
    'gradient_final_km', 'lag_rider_points_season',
    'lag_rider_rank_season',
    'lag_race_competitiveness_median',
    'lag_team_power_index'
]
X_all = df[features_benchmark]

# Gruppe 2 & 3: Die Zielvariablen und Gruppen
y_class_all = df['target_ordinal_relevance']
y_rank_all = df['rank']
stage_groups_all = df['stage_id']  # Wichtig für das Grouping

# Gruppe 4: Die Metadaten zur Auswertung (Fahrernamen, Teams etc.)
meta_cols = ['meta_year', 'meta_name', 'meta_current_team', 'meta_race', 'stage_nr', 'stage_id']
df_meta_all = df[meta_cols]
```

# Train Test Split


```python
# Definiere die Masken basierend auf den Saisons/Jahren
train_mask = df_meta_all['meta_year'] <= 2022
valid_mask = df_meta_all['meta_year'] == 2023
test_mask = df_meta_all['meta_year'] >= 2024

# 1. Splits für Features (X)
X_train = X_all[train_mask]
X_valid = X_all[valid_mask]
X_test = X_all[test_mask]

# 2. Splits für Klassifikations-Targets (y)
y_class_train = y_class_all[train_mask]
y_class_valid = y_class_all[valid_mask]
y_class_test = y_class_all[test_mask]

# 3. Splits für Regression / Ranking (y)
y_rank_train = y_rank_all[train_mask]
y_rank_valid = y_rank_all[valid_mask]
y_rank_test = y_rank_all[test_mask]

# 4. Splits für die Ranker-Gruppen (Wichtig für spätere Group-by-Operationen)
groups_train = stage_groups_all[train_mask]
groups_valid = stage_groups_all[valid_mask]
groups_test = stage_groups_all[test_mask]

# 5. Splits für die Metadaten (Sichert Namen und IDs für die Validierung und den Test)
meta_valid = df_meta_all[valid_mask].copy()
meta_test = df_meta_all[test_mask].copy()

# Kontroll-Ausgabe zur Verifikation in der Thesis
print("CHRONOLOGISCHES DATEN-SETUP (DREIFACH-SPLIT):")
print("------------------------------------------------------------------")
print(f"➔ Trainingsset (Saisons <= 2022) : {X_train.shape[0]:>6,d} Zeilen")
print(f"➔ Validationsset (Saison 2023)   : {X_valid.shape[0]:>6,d} Zeilen")
print(f"➔ Testset (Saisons 2024/2025)    : {X_test.shape[0]:>6,d} Zeilen")
print("------------------------------------------------------------------")
```

    CHRONOLOGISCHES DATEN-SETUP (DREIFACH-SPLIT):
    ------------------------------------------------------------------
    ➔ Trainingsset (Saisons <= 2022) : 169,349 Zeilen
    ➔ Validationsset (Saison 2023)   :  8,897 Zeilen
    ➔ Testset (Saisons 2024/2025)    : 17,802 Zeilen
    ------------------------------------------------------------------
    


```python

print(f"➔ Trainings-Set (Saisons <= 2022) : {X_train.shape[0]:>7,d} Zeilen")
print(f"➔ Validations-Set (Saison 2023)   : {X_valid.shape[0]:>7,d} Zeilen")
print(f"➔ Test-Set       (Saisons >= 2024) : {X_test.shape[0]:>7,d} Zeilen")
print("------------------------------------------------------------------")
print(f"➔ Metadaten-Set  (Zukunft ab 2024) : {meta_test.shape[0]:>7,d} Zeilen (Inklusive Fahrer- und Teamnamen!)")
print("------------------------------------------------------------------\n")
```

    ➔ Trainings-Set (Saisons <= 2022) : 169,349 Zeilen
    ➔ Validations-Set (Saison 2023)   :   8,897 Zeilen
    ➔ Test-Set       (Saisons >= 2024) :  17,802 Zeilen
    ------------------------------------------------------------------
    ➔ Metadaten-Set  (Zukunft ab 2024) :  17,802 Zeilen (Inklusive Fahrer- und Teamnamen!)
    ------------------------------------------------------------------
    
    

# Speichern der Modelldatengruppen


```python
pfad = '../../data/processed'

splits_to_save = {
    # 1. Trainings-Splits (Saisons <= 2022)
    'X_train.pkl': X_train,
    'y_class_train.pkl': y_class_train,
    'y_rank_train.pkl': y_rank_train,
    'groups_train.pkl': groups_train,

    # 2. Validierungs-Splits (Saison 2023 - NEU hinzugefügt!)
    'X_valid.pkl': X_valid,
    'y_class_valid.pkl': y_class_valid,
    'y_rank_valid.pkl': y_rank_valid,
    'groups_valid.pkl': groups_valid,
    'meta_valid.pkl': meta_valid,

    # 3. Test-Splits (Saisons >= 2024)
    'X_test.pkl': X_test,
    'y_class_test.pkl': y_class_test,
    'y_rank_test.pkl': y_rank_test,
    'groups_test.pkl': groups_test,
    'meta_test.pkl': meta_test
}

print("STARTE PICKLE-EXPORT FÜR CHRONOLOGISCHEN DREIFACH-SPLIT...")
print("------------------------------------------------------------------")

for file_name, data_object in splits_to_save.items():
    full_export_path = os.path.join(pfad, file_name)
    data_object.to_pickle(full_export_path)
    print(f"✅ Erfolgreich exportiert: {full_export_path}")

print("------------------------------------------------------------------")
print("🎉 Alle 14 Daten-Splits wurden fehlerfrei im Projektpfad gesichert!")
```

    STARTE PICKLE-EXPORT FÜR CHRONOLOGISCHEN DREIFACH-SPLIT...
    ------------------------------------------------------------------
    ✅ Erfolgreich exportiert: ../../data/processed\X_train.pkl
    ✅ Erfolgreich exportiert: ../../data/processed\y_class_train.pkl
    ✅ Erfolgreich exportiert: ../../data/processed\y_rank_train.pkl
    ✅ Erfolgreich exportiert: ../../data/processed\groups_train.pkl
    ✅ Erfolgreich exportiert: ../../data/processed\X_valid.pkl
    ✅ Erfolgreich exportiert: ../../data/processed\y_class_valid.pkl
    ✅ Erfolgreich exportiert: ../../data/processed\y_rank_valid.pkl
    ✅ Erfolgreich exportiert: ../../data/processed\groups_valid.pkl
    ✅ Erfolgreich exportiert: ../../data/processed\meta_valid.pkl
    ✅ Erfolgreich exportiert: ../../data/processed\X_test.pkl
    ✅ Erfolgreich exportiert: ../../data/processed\y_class_test.pkl
    ✅ Erfolgreich exportiert: ../../data/processed\y_rank_test.pkl
    ✅ Erfolgreich exportiert: ../../data/processed\groups_test.pkl
    ✅ Erfolgreich exportiert: ../../data/processed\meta_test.pkl
    ------------------------------------------------------------------
    🎉 Alle 14 Daten-Splits wurden fehlerfrei im Projektpfad gesichert!
    
