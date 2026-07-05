# _07_model_usage.py

import os
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path

def calculate_ndcg(true_ranks, pred_scores, k=10):
    """Berechnet den Normalized Discounted Cumulative Gain (NDCG) bei K.

    Höherer vorhergesagter Score = bessere Platzierung im Modell.
    """
    # Sortiere Indizes nach absteigendem Score
    order = np.argsort(pred_scores)[::-1][:k]
    true_ranks_at_k = np.array(true_ranks)[order]

    # Relevanz-Definition im Radsport-Ranking: Rel = 1 / realer_platz
    relevance = 1.0 / true_ranks_at_k

    # Discounted Cumulative Gain (DCG)
    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))

    # Ideal DCG (IDCG) - Das perfekte Ranking der im Feld verfügbaren Fahrer
    ideal_order = np.sort(true_ranks)[:k]
    ideal_relevance = 1.0 / ideal_order
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))

    return dcg / idcg if idcg > 0 else 0.0


def calculate_map_at_10(true_ranks, pred_scores):
    """Berechnet die Mean Average Precision (MAP@10) für eine Etappe.

    Nutzt den festen Cut-Off bei 10, da das sportliche Relevanz-Kriterium
    (die echte Spitzengruppe) per Definition exakt 10 Fahrer umfasst.
    """
    # Sortiere Indizes nach absteigendem Vorhersage-Score
    order = np.argsort(pred_scores)[::-1][:10]
    true_ranks_at_10 = np.array(true_ranks)[order]

    relevant_found = 0
    precision_sum = 0.0

    for idx, true_rank in enumerate(true_ranks_at_10):
        if true_rank <= 10:
            relevant_found += 1
            precision_sum += relevant_found / (idx + 1)

    # Mathematisch saubere Division durch die Konstante R = 10
    return precision_sum / 10.0


def run_master_inference(df_live_features, df_stage_meta):
    """Zentraler Master-Inferenz-Knoten für alle aktiven Modellwelten (EBM & TabPFN).

    - Prüft zeitabhängig (Heute >= Etappendatum + 1), ob das Rennen vorbei ist.
    - Gibt die Top-10-Prognosen je Modell aus (inkl. Real_Rang).
    - Berechnet NDCG@5/@10/@20, MAP@10 und Top-k Accuracies bei historischen Rennen.
    - Misst die Inferenzgeschwindigkeit in Millisekunden.
    """
    print("\n" + "=" * 100)
    print("SCHRITT 7: MASTER-INFERENZ & AKADEMISCHE EVALUIERUNG")
    print("=" * 100)

    # 1. METADATEN-EXTRAKTION & ZEITLICHE PRÜFUNG
    stage_date_str = str(df_stage_meta["date"].iloc[0]).strip()

    try:
        stage_date = pd.to_datetime(stage_date_str, format="%d %b %Y", errors="coerce")
        if pd.isna(stage_date):
            stage_date = pd.to_datetime(stage_date_str, errors="coerce")

        today = pd.to_datetime(time.strftime("%Y-%m-%d"))
        is_stage_finished = today >= (stage_date + pd.Timedelta(days=1))
    except Exception as e:
        print(f"⚠️ Hinweis beim Datum-Parsing: {e}. Schalte auf Live-Modus.")
        is_stage_finished = False
        stage_date = "Unbekannt"

    # Sauberer Header
    print(f"ETAPPE       : {df_stage_meta['stage_url'].iloc[0]}")
    print(f"DATUM        : {stage_date_str}")
    print(f"PROFIL       : {df_stage_meta['distance'].iloc[0]} km | {df_stage_meta['vertical_meters'].iloc[0]} hm | Finale: {df_stage_meta['gradient_final_km'].iloc[0]}%")

    # Name deiner Ergebnisspalte aus der Trainingsphase
    target_rank_col = "reales_etappen_ergebnis"

    # 🔍 PRÄZISE DIAGNOSE FÜR DEN STATUS-LOG
    if is_stage_finished:
        if target_rank_col in df_live_features.columns:
            true_ranks = df_live_features[target_rank_col].values
            print("STATUS       : Etappe beendet & Ergebnisspalte gefunden")
            has_ground_truth = True
        else:
            true_ranks = None
            print(f"STATUS       : Etappe ist vorbei ({stage_date_str}), aber Ergebnisspalte '{target_rank_col}' fehlt im Dataframe")
            has_ground_truth = False
    else:
        true_ranks = None
        print(f"STATUS       : Live-Inferenz (Etappe findet am {stage_date_str})")
        has_ground_truth = False

    print("-" * 100)

    # 2. FEATURE-ALIGNMENT & MODEL-SETTING
    # Exakte 17 Features in der trainierten mathematischen Reihenfolge
    feature_cols = [
        "distance", "vertical_meters", "stage_nr", "team_tier", "age_at_race",
        "rider_bmi", "wind_stability_index", "weather_temp_mean", "weather_temp_trend",
        "weather_rain_prob_mean", "weather_precipitation_mean", "weather_humidity_mean",
        "gradient_final_km", "lag_rider_points_season", "lag_rider_rank_season",
        "lag_race_competitiveness_median", "lag_team_power_index"
    ]

    X_live = df_live_features[feature_cols].copy()
    model_dir = "../../data/models"

    # Beide Modell-Ergebnisse liegen als .pkl-Dateien im Verzeichnis vor
    models_to_run = {
        "EBM": "ebm_best_binary_ensemble.pkl",
        "TabPFN": "tabpfn_final_results.pkl"
    }

    metrics_summary = []

    # 3. ITERATIVE MODELLEVALUIERUNG & LAUFZEITMESSUNG
    for model_name, model_file in models_to_run.items():
        base_dir = Path(model_dir)
        model_path = base_dir / model_file

        if not model_path.exists():
            print(f"⚠️ Datei für {model_name} nicht gefunden unter {model_path}. Überspringe.")
            continue

        start_time = time.time()

        try:
            # Beide Cache-Dateien sicher über das Standard-pickle-Modul laden
            with open(model_path, "rb") as f:
                loaded_model = pickle.load(f)

            if model_name == "EBM":
                # Frank & Hall Summen-Inferenz über die 3 gelernten Bins
                ebm_top5, ebm_top10, ebm_top20 = loaded_model
                probs_top5 = ebm_top5.predict_proba(X_live)[:, 1]
                probs_top10 = ebm_top10.predict_proba(X_live)[:, 1]
                probs_top20 = ebm_top20.predict_proba(X_live)[:, 1]
                pred_scores = probs_top5 + probs_top10 + probs_top20

            elif model_name == "TabPFN":
                # 1. Dynamischer ID-Sicherheitsfilter (Verhindert KeyError)
                if "stage_id" in df_stage_meta.columns:
                    current_stage_id = df_stage_meta["stage_id"].iloc[0]
                elif "stage_id" in df_live_features.columns:
                    current_stage_id = df_live_features["stage_id"].iloc[0]
                else:
                    # Fallback: Extrahiere und kodiere die ID direkt aus der procyclingstats-URL
                    stage_url = str(df_stage_meta["stage_url"].iloc[0])
                    url_parts = stage_url.split("/")
                    try:
                        race_name = url_parts[-3]
                        race_year = url_parts[-2]
                        stage_str = url_parts[-1].upper().replace("STAGE-", "ST")
                        current_stage_id = f"{race_name}_{race_year}_{stage_str}"
                    except Exception:
                        current_stage_id = loaded_model["stage_id"].iloc[0]

                # 2. Filtere die vorberechneten Scores für genau diese Etappe heraus
                df_stage_tabpfn = loaded_model[loaded_model["stage_id"] == current_stage_id]

                # Falls die generierte ID nicht im Cache existiert, weicher Teilstring-Match
                if df_stage_tabpfn.empty:
                    url_race_part = url_parts[-3] if len(url_parts) > 3 else "unbekannt"
                    df_stage_tabpfn = loaded_model[loaded_model["stage_id"].str.contains(url_race_part, na=False, case=False)]
                    if not df_stage_tabpfn.empty:
                        fallback_id = df_stage_tabpfn["stage_id"].iloc[0]
                        df_stage_tabpfn = loaded_model[loaded_model["stage_id"] == fallback_id]

                # 3. Robustes Spalten-Mapping (Unterstützt beide Namenskonventionen)
                score_col = "score_topk_raw_sum" if "score_topk_raw_sum" in df_stage_tabpfn.columns else "raw_prediction"
                pfn_map = dict(zip(df_stage_tabpfn["meta_name"], df_stage_tabpfn[score_col]))

                # Zuweisung der vorberechneten Scores auf das aktuelle Live-Feld
                pred_scores = df_live_features["name"].map(pfn_map).fillna(0.0).values

        except Exception as e:
            print(f"❌ Fehler bei Inferenz von {model_name}: {e}")
            continue

        elapsed_time = (time.time() - start_time) * 1000

        # Vorhersage-Struktur aufbauen
        df_result = df_live_features[["name", "team"]].copy()
        df_result["Score"] = pred_scores

        if has_ground_truth:
            df_result["Real_Rang"] = true_ranks
        else:
            df_result["Real_Rang"] = np.nan

        # Modell-Rangzuordnung (Höchster Score = Rang 1)
        df_result["Pred_Rang"] = (
            df_result["Score"].rank(method="first", ascending=False).astype(int)
        )
        df_top_10 = df_result.sort_values(by="Score", ascending=False).head(10)

        # ➔ OUTPUT A: TOP-10-PROGNOSE PRO MODELL IN DER KONSOLE
        print(f"\n➔ PROGNOSE MODELL: {model_name:<25} | Laufzeit: {elapsed_time:.2f} ms")
        print("-" * 110)

        df_output_print = df_top_10.copy()
        df_output_print["Real_Rang"] = df_output_print["Real_Rang"].fillna("N/A")
        print(
            df_output_print.to_string(
                index=False,
                columns=["Pred_Rang", "name", "team", "Score", "Real_Rang"],
            )
        )
        print("-" * 110)

        # OUTPUT B: METRIKEN BERECHNEN
        if has_ground_truth and true_ranks is not None:
            # Robustheits-Fix: Fehlende Ränge (NaNs) mit Penalty-Platz belegen, damit NDCG rechnet
            clean_true_ranks = pd.Series(true_ranks).fillna(999).values

            ndcg_5 = calculate_ndcg(clean_true_ranks, pred_scores, k=5)
            ndcg_10 = calculate_ndcg(clean_true_ranks, pred_scores, k=10)
            ndcg_20 = calculate_ndcg(clean_true_ranks, pred_scores, k=20)
            map_10 = calculate_map_at_10(clean_true_ranks, pred_scores)

            # Check auf den wahren Sieger (Real_Rang == 1)
            winner_rows = df_result[df_result["Real_Rang"] == 1]
            if not winner_rows.empty:
                true_winner = winner_rows["name"].iloc[0]
                predicted_order = (
                    df_result.sort_values(by="Score", ascending=False)["name"]
                    .tolist()
                )

                top1_acc = 1 if true_winner == predicted_order[0] else 0
                top5_acc = 1 if true_winner in predicted_order[:5] else 0
                top10_acc = 1 if true_winner in predicted_order[:10] else 0
                top20_acc = 1 if true_winner in predicted_order[:20] else 0
            else:
                top1_acc = top5_acc = top10_acc = top20_acc = 0

            metrics_summary.append(
                {
                    "Modell": model_name,
                    "Laufzeit (ms)": round(elapsed_time, 2),
                    "NDCG@5": round(ndcg_5, 4),
                    "NDCG@10": round(ndcg_10, 4),
                    "NDCG@20": round(ndcg_20, 4),
                    "MAP@10": round(map_10, 4),
                    "Top-1 Acc": f"{top1_acc*100}%",
                    "Top-5 Acc": f"{top5_acc*100}%",
                    "Top-10 Acc": f"{top10_acc*100}%",
                    "Top-20 Acc": f"{top20_acc*100}%",
                }
            )
        else:
            metrics_summary.append(
                {
                    "Modell": model_name,
                    "Laufzeit (ms)": round(elapsed_time, 2),
                    "NDCG@5": "N/A",
                    "NDCG@10": "N/A",
                    "NDCG@20": "N/A",
                    "MAP@10": "N/A",
                    "Top-1 Acc": "N/A",
                    "Top-5 Acc": "N/A",
                    "Top-10 Acc": "N/A",
                    "Top-20 Acc": "N/A",
                }
            )

    # 4. KONTRASTIERUNGS-REPORT (ZUSAMMENFASSUNG FÜR DEIN PROTOKOLL)
    print("\n" + "=" * 110)
    print("MODELL-VERGLEICH (ETAPPEN-REPORT)")
    print("=" * 110)
    df_metrics = pd.DataFrame(metrics_summary)
    print(df_metrics.to_string(index=False))
    print("=" * 110 + "\n")

    return df_metrics
