# _06_live_feature_builder.py

import os
import numpy as np
import pandas as pd


def prepare_live_features(df_final_report, df_stage_meta):
    """Konsolidiert alle Fahrer- und Etappendaten zu einer sauberen,

    unverschleierten Feature-Matrix und gibt Kontrollberichte aus.
    """
    if df_final_report.empty:
        print("-- Fehler: Keine Daten für die Feature-Generierung vorhanden.")
        return pd.DataFrame()

    print("\n" + "=" * 80)
    print(" SCHRITT 6: FEATURE-ENGINEERING & KONSOLIDIERUNG (LIVE-DATEN)")
    print("=" * 80)


    # 1. KONTROLL-AUSGABE: RENN- & ETAPPEN-METADATEN

    print("\nRENN-METADATEN (PROCYClINGSTATS):")
    print("-" * 80)
    for col in df_stage_meta.columns:
        print(f"  {col:<30}: {df_stage_meta[col].iloc[0]}")
    print("-" * 80)

    df_live = df_final_report.copy()


    # 2. FEATURE-INJEKTION (ETAPPEN- & WETTER-PROXIES)


    # Etappen-Spezifika
    df_live["distance"] = float(df_stage_meta["distance"].iloc[0])
    df_live["vertical_meters"] = float(df_stage_meta["vertical_meters"].iloc[0])
    df_live["stage_nr"] = int(df_stage_meta["stage_nr"].iloc[0])
    df_live["gradient_final_km"] = float(
        df_stage_meta["gradient_final_km"].iloc[0]
    )

    # Aktuelles Rennen aus den Metadaten auslesen für den Wetter-Proxy
    current_race = str(df_stage_meta["race"].iloc[0]).lower().strip()

    # Exakte historische Mediane injizieren je nach Grand Tour
    # Erweiterung um "tdf" für das Tour-de-France-Kürzel aus den Live-Metadaten
    if "giro" in current_race:
        df_live["weather_temp_mean"] = 18.33
        df_live["weather_humidity_mean"] = 59.25
        df_live["weather_rain_prob_mean"] = 0.040
        df_live["weather_precipitation_mean"] = 0.040
        df_live["wind_stability_index"] = 0.1990
        df_live["weather_temp_trend"] = -1.25
    elif "tour" in current_race or "tdf" in current_race:
        df_live["weather_temp_mean"] = 22.95
        df_live["weather_humidity_mean"] = 54.00
        df_live["weather_rain_prob_mean"] = 0.015
        df_live["weather_precipitation_mean"] = 0.015
        df_live["wind_stability_index"] = 0.1469
        df_live["weather_temp_trend"] = 0.12
    elif "vuelta" in current_race:
        df_live["weather_temp_mean"] = 24.24
        df_live["weather_humidity_mean"] = 50.50
        df_live["weather_rain_prob_mean"] = 0.000
        df_live["weather_precipitation_mean"] = 0.000
        df_live["wind_stability_index"] = 0.1736
        df_live["weather_temp_trend"] = -0.40
    else:
        # Globaler Fallback, falls ein unbekanntes Rennen übergeben wird
        df_live["weather_temp_mean"] = 21.5
        df_live["weather_humidity_mean"] = 55.0
        df_live["weather_rain_prob_mean"] = 0.01
        df_live["weather_precipitation_mean"] = 0.01
        df_live["wind_stability_index"] = 0.16
        df_live["weather_temp_trend"] = 0.0


    # 3. MATHEMATISCHE BERECHNUNGEN (ALTER & RENN-METRIKEN)

    # Taggenaues Alter am Renntag berechnen
    df_live["birthdate"] = pd.to_datetime(df_live["birthdate"])
    race_date = pd.to_datetime(df_stage_meta["date"].iloc[0])
    df_live["age_at_race"] = (
        race_date - df_live["birthdate"]
    ).dt.days / 365.25

    # Rennstärke (Median des Vorjahres-Rangs 'rank' aller Starter)
    df_live["lag_race_competitiveness_median"] = df_live["rank"].median()

    # Teamstärke (Median des Vorjahres-Rangs 'rank' gruppiert nach Team)
    team_medians = df_live.groupby("team")["rank"].median()
    df_live["lag_team_power_index"] = df_live["team"].map(team_medians)

    # Absicherung gegen NaNs bei den Rennmetriken
    df_live["lag_race_competitiveness_median"] = df_live[
        "lag_race_competitiveness_median"
    ].fillna(350.0)
    df_live["lag_team_power_index"] = df_live["lag_team_power_index"].fillna(
        350.0
    )


    # 4. WISSENSCHAFTLICHE BMI-IMPUTATION (GROUP BY NATIONALITÄT 'nat')

    # Sicherstellen, dass die BMI-Spalte numerisch ist
    df_live["bmi"] = pd.to_numeric(df_live["bmi"], errors="coerce")

    # Stufe 1: Länderspezifischen Median auf Basis der heutigen Startliste berechnen
    nat_bmi_medians = df_live.groupby("nat")["bmi"].transform("median")

    # Stufe 2: Fehlende Werte primär mit dem länderspezifischen Gruppen-Median füllen
    df_live["bmi"] = df_live["bmi"].fillna(nat_bmi_medians)

    # Stufe 3: Globaler Fallback (Median des gesamten heutigen Pelotons)
    global_field_bmi_median = df_live["bmi"].median()

    if pd.isna(global_field_bmi_median):
        global_field_bmi_median = 21.5

    df_live["bmi"] = df_live["bmi"].fillna(global_field_bmi_median)


    # 5. ALIGNMENT & RENAMING FÜR DIE DATEN-MATRIX

    tier_mapping = {
        "elite": 1,
        "pro": 2,
        "continental": 2
    }

    # Sicherstellen, dass Strings sauber in Kleinbuchstaben gemappt werden
    df_live["tier"] = df_live["tier"].astype(str).str.lower().str.strip()
    df_live["tier"] = df_live["tier"].map(tier_mapping).fillna(2)

    df_live = df_live.rename(
        columns={
            "points": "lag_rider_points_season",
            "rank": "lag_rider_rank_season",
            "tier": "team_tier",
            "bmi": "rider_bmi",
        }
    )


    # 6. EXAKTE REIHENFOLGE ERZWINGEN & DATAFRAME ERSTELLEN

    feature_cols = [
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

    # Erzeuge das Ausgabe-DataFrame mit Namen und Team für das Terminal
    display_cols = ["name", "team"] + feature_cols
    df_final_features = df_live[display_cols].copy()

    # Typ-Konvertierung für numerische Spalten erzwingen
    for col in feature_cols:
        df_final_features[col] = pd.to_numeric(
            df_final_features[col], errors="coerce"
        )


    # 7. UNGESCHNITTENE TERMINAL-AUSGABE DER FEATURE-MATRIX

    print("\nFINALE FEATURE-MATRIX FÜR DAS MODELL (ALLE 17 FEATURES):")
    print("-" * 120)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 1200)

    # formatierte Ausgabe der kompletten Startliste samt berechneter Werte
    print(df_final_features.to_string(index=False))
    print("-" * 120)
    print(
        f"Kontrolle: Matrix mit {len(df_final_features)} Fahrern und allen 17 Modell-Features erfolgreich generiert."
    )
    print("=" * 80)

    return df_final_features
