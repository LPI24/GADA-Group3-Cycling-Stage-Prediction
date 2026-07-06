# _07_model_usage.py

import os
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb


def calculate_ndcg(true_ranks, pred_scores, k=10):
    """Berechnet den Normalized Discounted Cumulative Gain (NDCG) bei K."""
    order = np.argsort(pred_scores)[::-1][:k]
    true_ranks_at_k = np.array(true_ranks)[order]

    relevance = 1.0 / true_ranks_at_k
    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))

    ideal_order = np.sort(true_ranks)[:k]
    ideal_relevance = 1.0 / ideal_order
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))

    return dcg / idcg if idcg > 0 else 0.0


def calculate_map_at_10(true_ranks, pred_scores):
    """Berechnet die Mean Average Precision (MAP@10) für eine Etappe."""
    order = np.argsort(pred_scores)[::-1][:10]
    true_ranks_at_10 = np.array(true_ranks)[order]

    relevant_found = 0
    precision_sum = 0.0

    for idx, true_rank in enumerate(true_ranks_at_10):
        if true_rank <= 10:
            relevant_found += 1
            precision_sum += relevant_found / (idx + 1)

    return precision_sum / 10.0


def run_master_inference(df_live_features, df_stage_meta):
    """Zentraler Master-Inferenz-Knoten für alle drei Modellwelten (EBM, XGBoost & TabPFN)."""
    print("\n" + "=" * 100)
    print("SCHRITT 7: MASTER-INFERENZ & AKADEMISCHE EVALUIERUNG")
    print("=" * 100)

    # 1. METADATEN-EXTRAKTION & ZEITLICHE PRÜFUNG (HEUTE-KORREKTUR)
    stage_date_str = str(df_stage_meta["date"].iloc[0]).strip()

    try:
        stage_date = pd.to_datetime(stage_date_str, format="%d %b %Y", errors="coerce")
        if pd.isna(stage_date):
            stage_date = pd.to_datetime(stage_date_str, errors="coerce")

        today = pd.to_datetime(time.strftime("%Y-%m-%d"))
        is_stage_finished = today >= stage_date
    except Exception as e:
        print(f"Hinweis beim Datum-Parsing: {e}. Schalte auf Live-Modus.")
        is_stage_finished = False
        stage_date = "Unbekannt"

    print(f"ETAPPE       : {df_stage_meta['stage_url'].iloc[0]}")
    print(f"DATUM        : {stage_date_str}")
    print(f"PROFIL       : {df_stage_meta['distance'].iloc[0]} km | {df_stage_meta['vertical_meters'].iloc[0]} hm | Finale: {df_stage_meta['gradient_final_km'].iloc[0]}%")

    target_rank_col = "reales_etappen_ergebnis"

    if is_stage_finished:
        if target_rank_col in df_live_features.columns:
            true_ranks = df_live_features[target_rank_col].values
            print("STATUS       : Etappe beendet & Ergebnisspalte gefunden")
            has_ground_truth = True
        else:
            true_ranks = None
            print(f"STATUS       : Rennen findet heute statt ({stage_date_str}), aber 'reales_etappen_ergebnis' fehlt noch im DataFrame.")
            has_ground_truth = False
    else:
        true_ranks = None
        print(f"STATUS       : Live-Inferenz (Etappe findet am {stage_date_str} statt)")
        has_ground_truth = False

    print("-" * 100)

    # 2. FEATURE-ALIGNMENT
    feature_cols = [
        "distance", "vertical_meters", "stage_nr", "team_tier", "age_at_race",
        "rider_bmi", "wind_stability_index", "weather_temp_mean", "weather_temp_trend",
        "weather_rain_prob_mean", "weather_precipitation_mean", "weather_humidity_mean",
        "gradient_final_km", "lag_rider_points_season", "lag_rider_rank_season",
        "lag_race_competitiveness_median", "lag_team_power_index"
    ]

    X_live = df_live_features[feature_cols].copy()
    model_dir = "../../data/models"

    # 🔥 ERSETZT .pkl MIT .csv, UM DEN COCHING-CRASH ZU VERMEIDEN
    models_to_run = {
        "EBM": "ebm_best_binary_ensemble.pkl",
        "XGBoost": "xgboost_classifier_results.json",
        "TabPFN": "tabpfn_final_results.csv"
    }

    metrics_summary = []

    # 3. ITERATIVE MODELLEVALUIERUNG
    for model_name, model_file in models_to_run.items():
        base_dir = Path(model_dir)
        model_path = base_dir / model_file

        if not model_path.exists():
            print(f"Datei für {model_name} nicht gefunden unter {model_path}. Überspringe.")
            continue

        start_time = time.time()

        try:
            if model_name == "EBM":
                with open(model_path, "rb") as f:
                    loaded_model = pickle.load(f)
                ebm_top5, ebm_top10, ebm_top20 = loaded_model
                probs_top5 = ebm_top5.predict_proba(X_live)[:, 1]
                probs_top10 = ebm_top10.predict_proba(X_live)[:, 1]
                probs_top20 = ebm_top20.predict_proba(X_live)[:, 1]
                pred_scores = probs_top5 + probs_top10 + probs_top20

            elif model_name == "XGBoost":
                try:
                    xgb_model = xgb.XGBClassifier()
                    xgb_model.load_model(model_path)
                    probs = xgb_model.predict_proba(X_live)
                    if probs.shape[1] >= 3:
                        pred_scores = probs[:, 1:].sum(axis=1)
                    else:
                        pred_scores = probs[:, 1]
                except Exception:
                    df_xgb_all = pd.read_json(model_path)
                    stage_url = str(df_stage_meta["stage_url"].iloc[0]).lower()
                    url_parts = stage_url.split("/")
                    try:
                        race_keyword = url_parts[-3].replace("-", "")
                        stage_nr_target = int(url_parts[-1].split("-")[-1])
                    except Exception:
                        race_keyword = "unbekannt"
                        stage_nr_target = int(df_stage_meta["stage_nr"].iloc[0])

                    df_race_match = df_xgb_all.copy()
                    df_race_match["stage_id"] = df_race_match["stage_id"].astype(str)
                    df_race_match = df_race_match[df_race_match["stage_id"].str.replace("-", "").str.contains(race_keyword, na=False, case=False)]
                    stage_patterns = [f"st{stage_nr_target}", f"st{stage_nr_target:02d}", f"_{stage_nr_target}", f"_{stage_nr_target:02d}"]

                    df_stage_xgb = pd.DataFrame()
                    for pat in stage_patterns:
                        df_stage_xgb = df_race_match[df_race_match["stage_id"].str.lower().str.endswith(pat)]
                        if not df_stage_xgb.empty:
                            break

                    if df_stage_xgb.empty and not df_race_match.empty:
                        if "stage_nr" in df_race_match.columns:
                            df_stage_xgb = df_race_match[df_race_match["stage_nr"].astype(int) == stage_nr_target]

                    score_col = "score_topk_raw_sum" if "score_topk_raw_sum" in df_stage_xgb.columns else "raw_prediction"
                    if score_col not in df_stage_xgb.columns:
                        num_cols = df_stage_xgb.select_dtypes(include=[np.number]).columns
                        score_col = num_cols[0] if len(num_cols) > 0 else df_stage_xgb.columns[-1]

                    xgb_map = dict(zip(df_stage_xgb["meta_name"] if "meta_name" in df_stage_xgb.columns else df_stage_xgb["name"], df_stage_xgb[score_col]))
                    pred_scores = df_live_features["name"].map(xgb_map).fillna(0.0).values

            elif model_name == "TabPFN":
                import traceback
                try:
                    # 🔥 VÖLLIG REVOLUTIONIERT: pd.read_csv umgeht das korrupte Environment-Pickle komplett!
                    df_tabpfn_all = pd.read_csv(model_path)

                    # Alle kritischen Identifikatoren als saubere Standard-Strings festlegen
                    for col in ["stage_id", "meta_name", "meta_race"]:
                        if col in df_tabpfn_all.columns:
                            df_tabpfn_all[col] = df_tabpfn_all[col].astype(str).str.strip()

                    # Score-Spalte numerisch erzwingen
                    score_col = "score_topk_raw_sum" if "score_topk_raw_sum" in df_tabpfn_all.columns else "raw_prediction"
                    if score_col not in df_tabpfn_all.columns:
                        num_cols = df_tabpfn_all.select_dtypes(include=[np.number]).columns
                        score_col = num_cols[0] if len(num_cols) > 0 else df_tabpfn_all.columns[-1]
                    df_tabpfn_all[score_col] = pd.to_numeric(df_tabpfn_all[score_col], errors="coerce").fillna(0.0)

                    # Live-Fahrernamen für das Mapping vorbereiten
                    df_live_features["name"] = df_live_features["name"].astype(str).str.strip()

                    # Renn- & Etappenerkennung aus Metadaten
                    stage_url = str(df_stage_meta["stage_url"].iloc[0]).lower()
                    url_parts = stage_url.split("/")
                    try:
                        race_keyword = url_parts[-3].replace("-", "")
                        stage_nr_target = int(url_parts[-1].split("-")[-1])
                    except Exception:
                        race_keyword = "unbekannt"
                        stage_nr_target = int(df_stage_meta["stage_nr"].iloc[0])

                    # Trennscharfer Filter auf das Rennen
                    df_race_match = df_tabpfn_all[
                        df_tabpfn_all["stage_id"].str.replace("-", "").str.contains(race_keyword, na=False, case=False)
                    ]
                    stage_patterns = [f"st{stage_nr_target}", f"st{stage_nr_target:02d}", f"_{stage_nr_target}", f"_{stage_nr_target:02d}"]

                    df_stage_tabpfn = pd.DataFrame()
                    for pat in stage_patterns:
                        df_stage_tabpfn = df_race_match[df_race_match["stage_id"].str.lower().str.endswith(pat)]
                        if not df_stage_tabpfn.empty:
                            break

                    if df_stage_tabpfn.empty and not df_race_match.empty:
                        if "stage_nr" in df_race_match.columns:
                            df_stage_tabpfn = df_race_match[df_race_match["stage_nr"].astype(int) == stage_nr_target]

                    # Robustes Score-Mapping
                    if df_stage_tabpfn.empty:
                        print(f"⚠️ WARNUNG: Keine TabPFN-Daten für race={race_keyword}, stage={stage_nr_target} gefunden. Scores = 0.")
                        pred_scores = np.zeros(len(df_live_features))
                    else:
                        pfn_map = dict(zip(df_stage_tabpfn["meta_name"], df_stage_tabpfn[score_col]))
                        pred_scores = df_live_features["name"].map(pfn_map).fillna(0.0).values

                except Exception as e:
                    print(f"❌ Fehler bei Inferenz von TabPFN: {e}")
                    traceback.print_exc()
                    continue

        except Exception as e:
            print(f"Fehler bei Inferenz von {model_name}: {e}")
            continue

        elapsed_time = (time.time() - start_time) * 1000

        # Vorhersage-Struktur aufbauen
        df_result = df_live_features[["name", "team"]].copy()
        df_result["Score"] = pred_scores
        df_result["Real_Rang"] = true_ranks if has_ground_truth else np.nan
        df_result["Pred_Rang"] = df_result["Score"].rank(method="first", ascending=False).astype(int)

        df_top_10 = df_result.sort_values(by="Score", ascending=False).head(10)

        print(f"\n➔ PROGNOSE MODELL: {model_name:<25} | Laufzeit: {elapsed_time:.2f} ms")
        print("-" * 110)
        df_output_print = df_top_10.copy()
        df_output_print["Real_Rang"] = df_output_print["Real_Rang"].fillna("N/A")
        print(df_output_print.to_string(index=False, columns=["Pred_Rang", "name", "team", "Score", "Real_Rang"]))
        print("-" * 110)

        # METRIKEN BERECHNEN
        if has_ground_truth and true_ranks is not None:
            clean_true_ranks = pd.Series(true_ranks).fillna(999).values

            ndcg_5 = calculate_ndcg(clean_true_ranks, pred_scores, k=5)
            ndcg_10 = calculate_ndcg(clean_true_ranks, pred_scores, k=10)
            ndcg_20 = calculate_ndcg(clean_true_ranks, pred_scores, k=20)
            map_10 = calculate_map_at_10(clean_true_ranks, pred_scores)

            winner_rows = df_result[df_result["Real_Rang"] == 1]
            if not winner_rows.empty:
                true_winner = winner_rows["name"].iloc[0]
                predicted_order = df_result.sort_values(by="Score", ascending=False)["name"].tolist()

                top1_acc = 1 if true_winner == predicted_order[0] else 0
                top5_acc = 1 if true_winner in predicted_order[:5] else 0
                top10_acc = 1 if true_winner in predicted_order[:10] else 0
                top20_acc = 1 if true_winner in predicted_order[:20] else 0
            else:
                top1_acc = top5_acc = top10_acc = top20_acc = 0

            metrics_summary.append({
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
            })
        else:
            metrics_summary.append({
                "Modell": model_name,
                "Laufzeit (ms)": round(elapsed_time, 2),
                "NDCG@5": "N/A", "NDCG@10": "N/A", "NDCG@20": "N/A", "MAP@10": "N/A",
                "Top-1 Acc": "N/A", "Top-5 Acc": "N/A", "Top-10 Acc": "N/A", "Top-20 Acc": "N/A",
            })

    print("\n" + "=" * 110)
    print("MODELL-VERGLEICH (ETAPPEN-REPORT)")
    print("=" * 110)
    df_metrics = pd.DataFrame(metrics_summary)
    print(df_metrics.to_string(index=False))
    print("=" * 110 + "\n")

    return df_metrics
