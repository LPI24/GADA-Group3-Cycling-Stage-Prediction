import joblib
import numpy as np
import pandas as pd

# ==============================================================================
# 1. MODELL-LOAD
# ==============================================================================

print("Lade die drei trainierten TabPFN Frank-&-Hall-Modelle...")

model_tabpfn_top5 = joblib.load("../../data/models/model_path_top5.joblib")
model_tabpfn_top10 = joblib.load("../../data/models/model_path_top10.joblib")
model_tabpfn_top20 = joblib.load("../../data/models/model_path_top20.joblib")

print("TabPFN-Modelle erfolgreich initialisiert.")

# ==============================================================================
# 2. FEATURE-VORBEREITUNG
# ==============================================================================

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
# 3. INFERENZ INNERHALB DES ETAPPEN-LOOPS
# ==============================================================================

X_explain = group[FEATURE_COLUMNS].copy()

print(
    f"Berechne TabPFN-Wahrscheinlichkeiten für Etappe {stage_id} ({len(X_explain)} Fahrer)..."
)

probs_tabpfn_top5 = model_tabpfn_top5.predict_proba(X_explain)[:, 1]
probs_tabpfn_top10 = model_tabpfn_top10.predict_proba(X_explain)[:, 1]
probs_tabpfn_top20 = model_tabpfn_top20.predict_proba(X_explain)[:, 1]

df_test_eval.loc[df_test_eval["stage_id"] == stage_id, "score_tabpfn"] = (
    probs_tabpfn_top5 + probs_tabpfn_top10 + probs_tabpfn_top20
)
