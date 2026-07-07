import joblib
import numpy as np
import pandas as pd
from tabpfn_client import TabPFNClient

# ==============================================================================
# 1. API-AUTHENTIFIZIERUNG & MODELL-LOAD
# ==============================================================================
# WICHTIG: Setze hier den echten API-Key aus dem PriorLabs-Konto deines Kollegen ein!
API_KEY = "tabpfn_sk_BleumEnMePsRs8ibsHCeyCKVwh-ZJjHnW05xHYjW5Fg"
TabPFNClient(api_key=API_KEY)

print("⏳ Lade die drei trainierten TabPFN Frank-&-Hall-Modelle...")
# Pfade zu den .joblib-Dateien deines Kollegen anpassen:
model_tabpfn_top5 = joblib.load("../../data/models/model_path_top5.joblib")
model_tabpfn_top10 = joblib.load("../../data/models/model_path_top10.joblib")
model_tabpfn_top20 = joblib.load("../../data/models/model_path_top20.joblib")
print("✅ TabPFN-Modelle erfolgreich initialisiert.")

# ==============================================================================
# 2. FEATURE-VORBEREITUNG (Aus deinem Notebook-Stand)
# ==============================================================================
# Die 17 Features müssen exakt in dieser Reihenfolge an TabPFN übergeben werden!
FEATURE_COLUMNS = [
    "distance",
    "vertical_meters",
    "stage_nr",
    "team_tier",
    "age_at_race",
    "rider_bmi",
    "wind_stability_index",
    "weather_temp_mean",
    "weather_temp_trend",
    "weather_rain_prob_mean",
    "weather_precipitation_mean",
    "weather_humidity_mean",
    "gradient_final_km",
    "lag_rider_points_season",
    "lag_rider_rank_season",
    "lag_race_competitiveness_median",
    "lag_team_power_index",
]

# ==============================================================================
# 3. LIVE-INFERENZ INNERHALB DEINES ETAPPEN-LOOPS
# ==============================================================================
# Angenommen, du loopst gerade durch deine Test-Etappen (2024/2025):
# `group` ist der gefilterte DataFrame für genau die aktuelle Etappe.

# Extrahiere die reinen Inferenz-Features in der korrekten Reihenfolge:
X_explain = group[FEATURE_COLUMNS].copy()

print(
    f"📡 Sende Request an PriorLabs Cloud für Etappe {stage_id} ({len(X_explain)} Fahrer)..."
)

# 1. Pointwise-Wahrscheinlichkeiten über die Cloud-Schnittstelle abfragen
# [:, 1] extrahiert wie bei deinen lokalen Modellen die Wahrscheinlichkeit für einen "Treffer"
probs_tabpfn_top5 = model_tabpfn_top5.predict_proba(X_explain)[:, 1]
probs_tabpfn_top10 = model_tabpfn_top10.predict_proba(X_explain)[:, 1]
probs_tabpfn_top20 = model_tabpfn_top20.predict_proba(X_explain)[:, 1]

# 2. Sequentielle Frank-&-Hall-Ordinalfusion für den finalen TabPFN-Ranking-Score
# Genau wie bei deiner EBM: Rohwahrscheinlichkeiten einfach addieren!
df_test_eval.loc[df_test_eval["stage_id"] == stage_id, "score_tabpfn"] = (
    probs_tabpfn_top5 + probs_tabpfn_top10 + probs_tabpfn_top20
)

# ==============================================================================
# 4. DIREKTER METRIK-CHECK (Integrierbar in deine Summary-Tabelle)
# ==============================================================================
# Jetzt existiert 'score_tabpfn' parallel zu 'score_baseline' und 'score_tuned'.
# Du kannst es nun exakt durch dieselben Evaluationsschleifen jagen:
# -> ndcg_score(y_true_relevance, scores_tabpfn, k=k)
# -> Mean Average Precision (MAP@10) für TabPFN berechnen
# -> Winner Hit Rates für TabPFN ableiten
