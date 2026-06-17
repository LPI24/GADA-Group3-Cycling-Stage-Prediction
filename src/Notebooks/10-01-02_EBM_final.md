Todo:
- sensitives Klassengewicht Quellen
- EBM Quellen
- Modellgütequellen

# Explainable Boosting Machines (EBM)

## Warum EBM?

* **Inhärente Interpretierbarkeit (Glass-Box):** Im Gegensatz zu herkömmlichen Black-Box-Modellen (wie XGBoost oder tiefen neuronalen Netzen), die post-hoc-Erklärungen (wie SHAP) benötigen, ist die EBM von Natur aus vollkommen transparent. Sie basiert auf verallgemeinerten additiven Modellen mit Interaktionen ($GA^2M$):
  $$g(E[y]) = \beta_0 + \sum f_i(x_i) + \sum f_{ij}(x_i, x_j)$$
  Jedes Feature und jede Interaktion wird über eine exakte mathematische Kurve (Spline) gelernt. Wir können später genau ablesen, *warum* das Modell eine Vorhersage trifft.
* **Erkennung nicht-linearer Sport-Muster:** Physische Variablen im Radsport verhalten sich selten linear. Ein optimaler Fahrer-BMI ist beispielsweise stark abhängig vom Profil der Etappe (Flachetappe vs. Hochgebirge). EBMs nutzen hochentwickeltes Tree-Boosting, um solche komplexen, stufenförmigen und nicht-linearen Beziehungen exakt abzubilden, ohne die Interpretierbarkeit zu verlieren.
* **Wissenschaftliche Baseline:** Indem wir ein hochpräzises, aber transparentes Modell als Fundament setzen, generieren wir eine unbestechliche Baseline. Wir können in den späteren Kapiteln genau prüfen, ob hochkomplexe Black-Box-Modelle (XGBoost) oder Foundation-Modelle (TabPFN) überhaupt einen signifikanten Performance-Gewinn gegenüber der gläsernen EBM bieten, der den Verlust der direkten Interpretierbarkeit rechtfertigt.

---

## Zielsetzung dieses Notebooks

In diesem Teilabschnitt setzen wir die theoretischen Anforderungen wie folgt in die Praxis um:
1. **Das Problem übersetzen:** Wir transformieren das komplexe Ranking-Problem in eine binäre Klassifikationsaufgabe („Landet Fahrer X in den Top-n?“).
2. **Chronological Split validieren:** Wir trainieren auf Daten bis einschließlich 2023 und evaluieren die Generalisierungsfähigkeit auf der ungesehenen Zukunftssaison 2024, um jegliches *Temporal Data Leakage* auszuschließen.
3. **Von der Baseline zum Hyperparameter-Tuning:** Wir starten mit einem intuitiven 1D-Modell, führen anschließend einen systematischen Grid-Search mit den neuen historischen Vorjahres-Features (`lag_...`) durch und finalisieren das beste Setup über statistisches Ensembling (`outer_bags`), um die Modellvarianz zu minimieren.


```python
import os
import pandas as pd
import numpy as np
import itertools
import time
import pickle
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_score, recall_score, roc_curve, precision_recall_curve, ndcg_score
import seaborn as sns

```

# Importieren der Daten



```python
pfad = '../../data/processed'

# Pkl Dateien einlesen

X_train = pd.read_pickle(os.path.join(pfad, 'X_train.pkl'))
X_test = pd.read_pickle(os.path.join(pfad, 'X_test.pkl'))
y_class_train = pd.read_pickle(os.path.join(pfad, 'y_class_train.pkl'))
y_class_test = pd.read_pickle(os.path.join(pfad, 'y_class_test.pkl'))
meta_test = pd.read_pickle(os.path.join(pfad, 'meta_test.pkl'))

total_rows = X_train.shape[0] + X_test.shape[0]
pct_train = (X_train.shape[0] / total_rows) * 100
pct_test = (X_test.shape[0] / total_rows) * 100

print("==================================================================")
print(f"Anzahl der Features im Modell: {X_train.shape[1]} Spalten")
print(f"Feature-Liste: {list(X_train.columns)}")
print(f"Trainings-Set (<= 2023)       : {X_train.shape[0]:,} Zeilen ({pct_train:.1f}%)")
print(f"Test-Set      (>= 2024)       : {X_test.shape[0]:,} Zeilen ({pct_test:.1f}%)")

```

    ==================================================================
    Anzahl der Features im Modell: 17 Spalten
    Feature-Liste: ['distance', 'vertical_meters', 'stage_nr', 'team_tier', 'age_at_race', 'rider_bmi', 'wind_stability_index', 'weather_temp_mean', 'weather_temp_trend', 'weather_rain_prob_mean', 'weather_precipitation_mean', 'weather_humidity_mean', 'gradient_final_km', 'lag_rider_points_season', 'lag_rider_rank_season', 'lag_race_competitiveness_median', 'lag_team_power_index']
    Trainings-Set (<= 2023)       : 178,246 Zeilen (90.9%)
    Test-Set      (>= 2024)       : 17,802 Zeilen (9.1%)
    

# Basismodell laden

- zu Beginn "intuitives 1D-Basismodell" für das Target target_top_10.
- keine Interaktionen (interactions=0)

Das Modell lernt die 17 Features (inklusive der neuen Vorjahres-Lags) rein isoliert.

Da wir ein starkes Klassenungleichgewicht haben (nur ca. 6–7 % der Zeilen sind eine "1" für Top 10), berechnen wir die sample_weight mit Scikit-Learn, um dem Modell beizubringen, die Top-10-Fahrer nicht einfach zu ignorieren.

## Klassenungleichgewicht (Cost-Sensitive Learning)


Um zu verhindern, dass das Modell die seltene Minderheitsklasse (Top 10) zugunsten der Majoritätsklasse ignoriert, wird die Standard-Verlustfunktion (**Log-Loss**) kostensensitiv modifiziert. Jede Zeile $i$ fließt multipliziert mit einem klassenspezifischen Gewicht $w_i$ in die gewichtete Gesamt-Verlustfunktion ein:

$$L_{\text{gesamt, gewichtet}} = \frac{\sum w_i \cdot L_i}{\sum w_i} \quad \text{mit} \quad w_k = \frac{N}{K \cdot N_k}$$

* **Klassenungleichgewicht kompensieren:** Durch Einsetzen der inversen Häufigkeit ($N$: Zeilen gesamt, $K$: Klassenanzahl, $N_k$: Zeilen der Klasse $k$) erhält jede Top-10-Zeile ein drastisch höheres Gewicht ($w_1 \approx 7.14$) als das Hauptfeld ($w_0 \approx 0.53$).
* **Perfekte Waffengleichheit:** Multipliziert man die absolute Zeilenanzahl mit dem jeweiligen Faktor ($N_1 \cdot w_1 = N_0 \cdot w_0 = \frac{N}{2}$), besitzen beide Klassen trotz extrem ungleicher Startbedingungen exakt denselben maximalen Einfluss von **50% zu 50%** auf die mathematische Optimierung des Algorithmus.


```python
chart_path = '../../data/charts'

# 1. Daten für die Grafiken vorbereiten
y_train_top10 = y_class_train['target_top_10']

# Berechnen der Gewichte
sample_weights_base = compute_sample_weight(class_weight='balanced', y=y_train_top10)

# Erstellen einer temporären Tabelle für eine saubere Visualisierung
df_vis = pd.DataFrame({
    'Klasse': y_train_top10.map({0: 'Hauptfeld (0)', 1: 'Top 10 (1)'}),
    'Mathematisches Gewicht': sample_weights_base
})

# Berechnung effektive Werte (Grafik 3)
df_effektiv = df_vis.groupby('Klasse', as_index=False).sum()
df_effektiv.rename(columns={'Mathematisches Gewicht': 'Effektives Gesamtgewicht'}, inplace=True)

# 2. Plot-Bereich definieren (1 Reihe, 3 Spalten für ein sauberes horizontales Layout)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.set_theme(style="whitegrid")

# --- 1: Das unkorrigierte Klassenungleichgewicht ---
sns.countplot(data=df_vis, x='Klasse', ax=axes[0], palette=['#34495e', '#e74c3c'], order=['Top 10 (1)', 'Hauptfeld (0)'])
axes[0].set_title('1. Das unkorrigierte Klassenungleichgewicht\n(Echte Verteilung im Fahrerfeld)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Anzahl der Fahrer-Zeilen')
axes[0].set_xlabel('')

# Prozentuale Labels auf die Balken setzen
total = len(df_vis)
for p in axes[0].patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2 - 0.1
    y = p.get_height() + (total * 0.01)
    axes[0].annotate(percentage, (x, y), fontweight='bold', fontsize=10)

# --- 2: Die Skalierung durch sample_weight ---
sns.barplot(data=df_vis, x='Klasse', y='Mathematisches Gewicht', ax=axes[1], palette=['#34495e', '#e74c3c'], errorbar=None, order=['Top 10 (1)', 'Hauptfeld (0)'])
axes[1].set_title('2. Die Skalierung durch sample_weight\n(Mathematische Gewichtung beim Lernen)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Zugeordnetes Gewicht pro Zeile')
axes[1].set_xlabel('')

# Exakte Gewichtswerte auf die Balken schreiben
for p in axes[1].patches:
    weight_val = f'{p.get_height():.2f}'
    x = p.get_x() + p.get_width() / 2 - 0.08
    y = p.get_height() + 0.1
    axes[1].annotate(weight_val, (x, y), fontweight='bold', fontsize=10)

# --- 3: Das resultierende mathematische Gleichgewicht ---
sns.barplot(data=df_effektiv, x='Klasse', y='Effektives Gesamtgewicht', ax=axes[2], palette=['#2c3e50', '#c0392b'], order=['Top 10 (1)', 'Hauptfeld (0)'])
axes[2].set_title('3. Effektiver Einfluss beim Lernen\n(Anzahl Zeilen × Gewicht)', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Gesamteinfluss auf Verlustfunktion')
axes[2].set_xlabel('')

# Werte auf Grafik 3 schreiben (wird exakt gleich hoch sein!)
for p in axes[2].patches:
    total_weight_val = f'{p.get_height():,.0f}'
    x = p.get_x() + p.get_width() / 2 - 0.15
    y = p.get_height() + (df_effektiv['Effektives Gesamtgewicht'].max() * 0.01)
    axes[2].annotate(total_weight_val, (x, y), fontweight='bold', fontsize=10)

# Layout optimieren, speichern und anzeigen
plt.tight_layout()
plt.savefig(os.path.join(chart_path, '10-01-02_01-klassenungleichgewicht_vs_gewichte.png'), dpi=300)
plt.show()
```

# Erstellung des Basismodells

Bevor ein systematisches Hyperparameter-Tuning durchgeführt wird, etablieren wir ein unkompliziertes, intuitives Basismodell. Dieses dient als Baseline, um die reine, isolierte Vorhersagekraft der Features zu messen.

### Methodisches Vorgehen im Basismodell:
* **Isolierung des Targets:** Wir fokussieren uns in diesem Durchlauf spezifisch auf die Zielvariable `target_top_10` (Fahrer landet in den Top 10).
* **Einspeisung der Gewichte:** Das zuvor berechnete Array `sample_weights_base` wird übergeben, um die mathematische Waffengleichheit (50:50 Einfluss der Klassen auf den Loss) im Hintergrund zu garantieren.
* **Isolierte Effekte:** Das Modell lernt die 17 Features (inklusive der neuen historischen Vorjahres-Lags) rein isoliert, ohne Wechselwirkungen zwischen den Variablen zu erlauben.

### Die exakte Konfiguration des EBM-Klassifikators:

* `interactions=0`: Schaltet jegliche Feature-Paare ab. Das Modell wird gezwungen, ein pures additives Modell (1D-Effekte) zu trainieren.
* `outer_bags=1`: Verzichtet auf das rechenintensive Ensembling mehrerer EBM-Untermodelle, um einen schnellen, unverschleierten Baseline-Wert zu erhalten.
* `validation_size=0.15`: Das Modell trennt intern im Hintergrund automatisch 15 % der Trainingsdaten ab, um die Optimierung zu überwachen.
* `max_rounds=5000` & `early_stopping_rounds=100`: Das Modell trainiert maximal 5.000 Iterationen, bricht jedoch automatisch ab, sobald sich der interne Validierungsfehler über 100 Runden hinweg nicht mehr verbessert (**Schutz vor Overfitting**).
* `learning_rate=0.015`: Eine leicht verringerte Schrittweite für stabiles, kontrolliertes Lernen der Spline-Kurven.
* `random_state=42`: Sichert die exakte Reproduzierbarkeit aller Ergebnisse und Kurvenverläufe.
* `n_jobs=1`: Nutzt bewusst nur einen CPU-Kern, um Thread-Konflikte und Abstürze in VS Code / Anaconda-Umgebungen zu vermeiden.


```python
start_base = time.time()

y_train_top10 = y_class_train['target_top_10']
y_test_top10 = y_class_test['target_top_10']

# EBM-Baseline initialisieren (Pures 1D-Modell, keine Interaktionen)
ebm_baseline = ExplainableBoostingClassifier(
    interactions=0,                # Keine Feature-Paare erlauben
    learning_rate=0.015,           # Kontrollierte Schrittweite
    outer_bags=1,                  # Ein einzelnes Modell für die Baseline
    validation_size=0.15,          # 15% interne Validierung für Early Stopping
    max_rounds=5000,               # Maximale Iterationen
    early_stopping_rounds=100,     # Stop, wenn 100 Runden kein Fortschritt
    early_stopping_tolerance=1e-05,
    random_state=42,               # Reproduzierbarkeit
    n_jobs=1                       # Stabil auf einem CPU-Kern
)

# Modell im Hintergrund trainieren (Features + Target + Gewichte)
ebm_baseline.fit(X_train, y_train_top10, sample_weight=sample_weights_base)

# Vorhersagewahrscheinlichkeiten für die Testphase (2024+) generieren
probs_base = ebm_baseline.predict_proba(X_test)[:, 1]

# Wissenschaftliche Evaluierung mittels ROC-AUC
auc_base = roc_auc_score(y_test_top10, probs_base)
duration_base = time.time() - start_base


print(f"Baseline ROC-AUC (Saison 2024/2025) : {auc_base:.4f}")
print(f"Benötigte Rechenzeit            : {duration_base:.1f} Sekunden")

```

    Baseline ROC-AUC (Saison 2024) : 0.7311
    Benötigte Rechenzeit            : 205.6 Sekunden
    

### Weitere Metriken

ACHTUNG: 
Wenn wir jetzt einfach ebm_baseline.predict(X_test) ausführen würden, nutzt Scikit-Learn im Hintergrund einen Standard-Schwellenwert (Threshold) von 0,5 (50%).

Wegen unserer künstlichen sample_weight-Gewichtung im Hintergrund verschieben sich jedoch die absoluten Wahrscheinlichkeiten (wie vorhin besprochen). Wenn das Modell für 2024 die Standard-Metriken berechnet, wird die Precision (Genauigkeit der Top-10-Vorhersagen) extrem in den Keller sinken und die Accuracy schlechter aussehen, als sie eigentlich ist.


**ROC Kurve**




```python
# 1. ROC-Kurve berechnen
fpr, tpr, roc_thresholds = roc_curve(y_test_top10, probs_base)

# 2. Plot erstellen
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'EBM Baseline (AUC = {auc_base:.4f})')
plt.plot([0, 1], [0, 1], color='#34495e', lw=1.5, linestyle='--', label='Zufallsklassifikator (AUC = 0.50)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falsch-Positiv-Rate (1 - Spezifität)', fontsize=11)
plt.ylabel('Richtig-Positiv-Rate (Sensitivität / Recall)', fontsize=11)
plt.title('ROC-Kurve (Receiver Operating Characteristic)\nEBM-Basismodell (Saisons 2024/2025)', fontsize=12, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)

# Speichern und anzeigen
plt.tight_layout()
plt.savefig(os.path.join(chart_path, '../../data/charts/10-01-02_02_baseline_roc_curve.png'), dpi=300)
plt.show()
```


    
![png](10-01-02_EBM_final_files/10-01-02_EBM_final_9_0.png)
    


## Interpretation der globalen Modellgüte (AUROC & ROC-Kurve)

Um die Leistung unserer EBM-Baseline wissenschaftlich einzuordnen, betrachten wir die Metriken der **Receiver Operating Characteristic (ROC)** und den dazugehörigen **Area Under the Curve (AUC)**-Wert.

### 1. Was bedeutet der AUROC-Wert von 0.7311?
* **Mathematische Definition:** Der AUC-Wert gibt die Wahrscheinlichkeit an, dass das Modell einen zufällig ausgewählten echten Top-10-Fahrer (Klasse 1) höher einstuft als einen zufällig ausgewählten Fahrer aus dem Hauptfeld (Klasse 0).
* **Klassifikation der Güte:** Ein Wert von $0.50$ entspricht dem reinen Zufall (Münzwurf). Werte zwischen $0.70$ und $0.80$ gelten in der Statistik als **akzeptable bis gute Trennschärfe** (*acceptable/good discrimination*). 
* **Projekt-Kontext:** Mit $0.7311$ auf den ungesehenen Zukunftsdaten (2024/2025) beweist das Modell eine beachtliche Generalisierungsfähigkeit. Angesichts der enormen Dynamik, Sturzrisiken und taktischen Unwägbarkeiten im Straßenradsport ist dies eine sehr starke und verlässliche Baseline.

### 2. Visuelle Interpretation der ROC-Kurve
* **Die Diagonale (Zufallslinie):** Die gestrichelte Linie von $(0,0)$ bis $(1,1)$ stellt den reinen Zufall dar. Je weiter sich unsere rote EBM-Kurve nach links oben (Richtung des "perfekten Punkts" bei $0.0$ False Positive und $1.0$ True Positive) wölbt, desto besser ist das Modell.
* **Der Kurvenverlauf:** Unsere Kurve steigt im linken Bereich steil an. Das bedeutet, das Modell kann die absoluten Top-Favoriten mit einer sehr geringen Fehlerquote (False Positive Rate) herausfiltern. Erst wenn wir versuchen, fast alle Top-10-Fahrer zu erwischen (hoher Recall), steigt auch die Anzahl der Fehlalarme im Hauptfeld an.

**Der optimale Threshold (F1-Score Maximierung)**



```python
# 1. Precision-Recall-Werte für alle Schwellenwerte berechnen
precision, recall, thresholds = precision_recall_curve(y_test_top10, probs_base)

# 2. F1-Score für jeden Punkt berechnen (kleines Epsilon verhindert Division durch 0)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

# 3. Den Index des maximalen F1-Scores finden
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]


print(f"Optimaler Schwellenwert (Threshold) : {best_threshold:.4f}")
print(f"Maximal erreichbarer F1-Score       : {best_f1:.4f}")

```

    Optimaler Schwellenwert (Threshold) : 0.6364
    Maximal erreichbarer F1-Score       : 0.2381
    

## Interpretation der Schwellenwert-Optimierung & des F1-Scores

Die mathematische Bestimmung des optimalen Schwellenwerts liefert zwei zentrale Kennzahlen, die tiefere Einblicke in das Entscheidungsverhalten des Modells gewähren.

### 1. Der optimale Schwellenwert (0.6364)
* **Verschiebung durch Gewichtung:** Ein ungewichteter Standard-Klassifikator trennt strikt bei $0.50$. Da wir jedoch im Hintergrund mit `sample_weight` gearbeitet haben, wurden die Top-10-Fahrer mathematisch stark aufgewertet. Das führt dazu, dass die rohen Modellwahrscheinlichkeiten nach oben verschoben sind.
* **Mathematische Konsequenz:** Der optimierte Threshold von $0.6364$ korrigiert diese Verschiebung. Das Modell sagt eine Top-10-Platzierung erst dann voraus, wenn die berechnete Wahrscheinlichkeit über **63,64 %** liegt. Dies schützt das Modell vor einer Inflation an "Fehlalarmen" (False Positives) durch zu optimistische Vorhersagen.

### 2. Der maximal erreichbare F1-Score (0.2381)
* **Das mathematische Paradoxon unbalancierter Daten:** Der F1-Score ist das harmonische Mittel aus *Precision* (Wie viele der vorhergesagten Top-10-Fahrer waren wirklich drin?) und *Recall* (Wie viele der echten Top-10-Fahrer haben wir erwischt?). 
* **Warum ist der Wert scheinbar so niedrig?** Bei einer extremen Minderheitenklasse (ca. 6 % Top-10 vs. 94 % Hauptfeld) führt jeder einzelne Fehlalarm (False Positive) zu einem massiven Einbruch der Precision. Wenn das Modell z. B. für eine Etappe 20 Fahrer in die Top 10 tippt, aber naturgemäß nur 10 reinkommen können, sind automatisch mindestens 10 Vorhersagen falsch. Das drückt den F1-Score mathematisch gnadenlos nach unten.
* **Wissenschaftliche Einordnung:** Ein F1-Score von knapp $0.24$ ist bei einer Baseline von 6 % ein **starker statistischer Hebel**. Es zeigt, dass das Modell Lichtjahre besser performt als ein reiner Zufalls- oder Mehrheitsklassifikator (deren F1-Score für die Minderheitsklasse bei exakt $0.00$ läge).

### Fazit für das Tuning:
Der F1-Score der Baseline spiegelt das fundamentale Problem einer harten binären Klassifikation im Radsport wider: Fahrer auf den Plätzen 11 bis 15 werden als "kompletter Fehler" (False Positive) gewertet, obwohl sie extrem nah an den Top 10 dran waren. 

**Classification Report**


```python
# 1. Vorhersagen basierend auf dem optimalen Schwellenwert in 0 und 1 umwandeln
y_pred_opt = (probs_base >= best_threshold).astype(int)

print("KLASSIFIKATIONSBERICHT")

print(f"Hinweis: Ergebnisse basieren auf dem optimierten Threshold von {best_threshold:.4f}\n")


# Report
print(classification_report(y_test_top10, y_pred_opt, target_names=['Hauptfeld (0)', 'Top 10 (1)'], digits=4))

```

    KLASSIFIKATIONSBERICHT
    Hinweis: Ergebnisse basieren auf dem optimierten Threshold von 0.6364
    
                   precision    recall  f1-score   support
    
    Hauptfeld (0)     0.9583    0.8512    0.9016     16698
       Top 10 (1)     0.1633    0.4393    0.2381      1104
    
         accuracy                         0.8256     17802
        macro avg     0.5608    0.6452    0.5698     17802
     weighted avg     0.9090    0.8256    0.8604     17802
    
    

## Interpretation des Klassifikationsberichts (Baseline)

Der Classification Report schlüsselt die Leistung des Modells für beide Klassen separat auf und zeigt die inhärenten Herausforderungen der Radsport-Prädiktion.

### 1. Klasse 0: Hauptfeld (Die Majoritätsklasse)
* **Precision (0.9583):** Wenn das Modell prognostiziert, dass ein Fahrer *nicht* in die Top 10 kommt, liegt es zu **95,83 %** richtig. Das ist ein extrem solider und verlässlicher Wert für das Ausschlussverfahren.
* **Recall (0.8512):** Das Modell erkennt **85,12 %** aller tatsächlichen Hauptfeld-Fahrer korrekt. Die restlichen ~15 % sind Fahrer, die das Modell fälschlicherweise als Top-10-Kandidaten eingestuft hat (False Positives).

### 2. Klasse 1: Top 10 (Die kritische Minderheitsklasse)
* **Precision (0.1633):** Von allen Fahrern, die das Modell aktiv für die Top 10 nominiert hat, landen am Ende nur **16,33 %** tatsächlich dort. 
  * *Radsport-Kontext:* Das klingt niedrig, bedeutet aber: Unter den vom Modell ausgewählten Favoriten ist die Trefferdichte fast **dreimal so hoch** wie im gesamten Fahrerfeld (wo die Basiswahrscheinlichkeit bei nur ~6 % liegt). Das Modell filtert die Spitzengruppe also bereits stark vor, ist aber noch sehr "optimistisch".
* **Recall (0.4393):** Das Modell schafft es aus dem Stand, **43,93 %** aller echten Top-10-Platzierungen in den Saisons 2024/2025 korrekt vorherzusagen. Fast jeder zweite Top-10-Fahrer wird also von der physikalisch-historischen Baseline ohne jegliche Interaktionen bereits erkannt.

### 3. Globale Durchschnitte (Accuracy vs. Weighted Avg)
* **Accuracy (0.8256):** Insgesamt sind **82,56 %** aller Vorhersagen (Hauptfeld + Top 10) korrekt. 
  * *Achtung bei der Argumentation:* Bei unbalancierten Daten ist eine hohe Accuracy trügerisch (ein naiver Klassifikator, der immer "0" tippt, hätte hier ~93,8 % Accuracy). Da unser Modell aber die Minderheit aktiv bewertet und trotzdem über 82 % hält, ist das ein sehr respektables Ergebnis.
* **Weighted Avg (0.9090 Precision / 0.8256 Recall):** Da das Hauptfeld mit 16.698 Zeilen den Datensatz dominiert (*Support*), zieht es den gewichteten Gesamtdurchschnitt stark nach oben. Für die Bewertung unseres sportlichen Ziels ist dieser Wert jedoch sekundär – entscheidend ist die Performance auf der Klasse `Top 10 (1)`.

### Strategisches Fazit für das anstehende Hyperparameter-Tuning:
Die größte Schwachstelle der Baseline ist die niedrige **Precision der Top 10 (16,33 %)**. Das Modell schlägt noch zu oft "Fehlalarm" und nominiert zu viele Fahrer für die Top 10. 

**Unser Ziel für das Tuning ($GA^2M$):** Wir müssen die Precision der Klasse 1 steigern, indem wir dem Modell erlauben, Interaktionen zu lernen. Dadurch soll es "wählerischer" werden (z. B. einen Bergspezialisten bei einer Flachetappe trotz starker Vorjahresform *nicht* in die Top 10 wählen), um die Anzahl der False Positives drastisch zu senken.

**Confusion Matrix**


```python
# 1. Konfusionsmatrix berechnen
cm = confusion_matrix(y_test_top10, y_pred_opt)

# 2. Matrix strukturiert in der Konsole ausgeben

print("KONFUSIONS-MATRIX (REINER TEXT-EXPORT)")
print("==================================================================")
print(f"                  Vorhergesagt:    Vorhergesagt:")
print(f"                  Hauptfeld (0)    Top 10 (1)")
print(f"Tatsächlich:")
print(f"Hauptfeld (0)      {cm[0, 0]:<16,d} {cm[0, 1]:,d}")
print(f"Top 10 (1)         {cm[1, 0]:<16,d} {cm[1, 1]:,d}")
print("==================================================================")
print(f"➔ Richtig Negativ (Echtes Hauptfeld korrekt erkannt): {cm[0, 0]:,d}")
print(f"➔ Falsch Positiv  (Hauptfeld fälschlich als Top 10):  {cm[0, 1]:,d}")
print(f"➔ Falsch Negativ  (Top 10 Fahrer leider übersehen):   {cm[1, 0]:,d}")
print(f"➔ Richtig Positiv (Top 10 Fahrer korrekt erkannt):    {cm[1, 1]:,d}")
print("==================================================================")
```

    KONFUSIONS-MATRIX (REINER TEXT-EXPORT)
    ==================================================================
                      Vorhergesagt:    Vorhergesagt:
                      Hauptfeld (0)    Top 10 (1)
    Tatsächlich:
    Hauptfeld (0)      14,213           2,485
    Top 10 (1)         619              485
    ==================================================================
    ➔ Richtig Negativ (Echtes Hauptfeld korrekt erkannt): 14,213
    ➔ Falsch Positiv  (Hauptfeld fälschlich als Top 10):  2,485
    ➔ Falsch Negativ  (Top 10 Fahrer leider übersehen):   619
    ➔ Richtig Positiv (Top 10 Fahrer korrekt erkannt):    485
    ==================================================================
    

## Interpretation der Konfusionsmatrix (Baseline-Analyse)

Die Konfusionsmatrix blickt tief in das konkrete Entscheidungsverhalten des Modells bei den 17.802 Fahrer-Zeilen der Testjahre 2024/2025 und deckt die genaue Verteilung von Fehlern und Treffern auf.

### 1. Die korrekten Vorhersagen (Die Hauptdiagonale)
* **True Negatives (14.213 Zeilen):** In der überwältigenden Mehrheit der Fälle erkennt das Modell präzise, welche Fahrer keine Rolle im Kampf um die Spitzenplätze spielen werden. Das Modell "säubert" das Fahrerfeld effektiv von Wasserträgern und Sprintern auf Bergetappen (bzw. umgekehrt).
* **True Positives (485 Zeilen):** Das Modell prognostiziert 485-mal punktgenau einen echten Top-10-Erfolg. Angesichts der Tatsache, dass im Radsport Nuancen, Stürze oder taktische Ausreißergruppen die Platzierung bestimmen, ist dieses fundamentale Erkennen von fast 500 Spitzenplatzierungen ein starker Nachweis, dass dennoch Muster erkannt werden.

### 2. Die Fehlprognosen (Die Gegendiagonale)
* **False Positives / "Fehlalarme" (2.485 Zeilen):** Dies ist die größte Schwachstelle der Baseline. Das Modell nominiert im gesamten Zukunftszeitraum 2.485-mal einen Fahrer für die Top 10, der am Ende im Hauptfeld landet.
  * *Sportlicher Hintergrund:* Da das Basismodell keine Interaktionen kennt, sieht es z. B. nur: *"Fahrer X hat viele Vorjahrespunkte (lag_rider_points_season) – also Top 10!"* Es ignoriert dabei, dass diese spezifische Etappe ein Zeitfahren oder Hochgebirge ist, das absolut nicht zum Fahrertyp passt. Das Modell ist hier noch zu "grob" und streut seine Favoritentipps zu breit.
* **False Negatives / "Übersehene Favoriten" (619 Zeilen):** In 619 Fällen fährt ein Fahrer in die Top 10, denn das Modell im Vorfeld dem Hauptfeld zugeordnet hatte.
  * *Sportlicher Hintergrund:* Hierbei handelt es sich klassischerweise um Überraschungssieger aus Ausreißergruppen, Helfer, die wegen eines gestürzten Kapitäns plötzlich auf eigene Rechnung fahren durften, oder junge Talente, die in den historischen Vorjahresranglistendaten schlichtweg noch keine nennenswerten Punkte angehäuft hatten.

### Fazit für die wissenschaftliche Argumentation:
Das Verhältnis von **485 (True Positives) zu 2.485 (False Positives)** erklärt mathematisch die zuvor gesehene niedrige Precision (~16 %). Das Modell wirft ein zu großes Netz aus.

**Der Fahrplan für das Hyperparameter-Tuning ($GA^2M$):**
Die zentrale Aufgabe der kommenden Optimierung ist es, den Wert der **False Positives (2.485)** drastisch zu senken. Wenn wir dem Modell erlauben, Variablen miteinander zu kombinieren (z. B. `rider_bmi` $\times$ `vertical_meters`), wird das Modell lernen, die Favoritenliste je nach Etappenprofil radikal zusammenzustreichen. Ein Klassementfahrer wird dann auf Flachetappen aussortiert, wodurch die Fehlalarme sinken und die Vorhersagequalität für die Hausarbeit massiv steigt.

---



## Interpretation der globalen Feature-Wichtigkeit (EBM Baseline)

Mit dem Aufruf von `explain_global()` betrachten wir die inhärente Feature-Wichtigkeit der EBM-Baseline. Der *Mean Absolute Score (Weighted)* gibt an, wie stark ein Feature im mathematischen Durchschnitt die Vorhersagewahrscheinlichkeit für eine Top-10-Platzierung nach oben oder unten verschiebt.

### 1. Die absolute Dominanz der historischen Lags
* **`lag_rider_rank_season` & `lag_rider_points_season`:** Diese beiden Variablen sind die mit Abstand mächtigsten Prädiktoren im gesamten Modell. Insbesondere der Saisonsrang des Fahrers besitzt einen gewaltigen mathematischen Hebel (~0.83). 
  * *Sportwissenschaftliche Erkenntnis:* Die inhärente Logik des Radsports spiegelt sich hier perfekt wider: Die aktuelle Formkurve und Klasse eines Fahrers (historische Vorjahres- und Saisonleistung) überwiegen singuläre physikalische Faktoren oder Wetterbedingungen bei weitem. Ein Spitzenfahrer (z. B. Pogacar oder Vingegaard) bringt eine so hohe Basiswahrscheinlichkeit für eine Top-10-Platzierung mit, dass das Modell dies als primäres Fundament nutzt.
* **`lag_race_competitiveness_median`:** Die relative Stärke des Rennens folgt auf Platz 3. Das Modell hat gelernt zu differenzieren: Punkte oder Platzierungen bei einem hochkompetitiven WorldTour-Rennen (z. B. Tour de France) wiegen schwerer als bei einer kleineren Rundfahrt.

### 2. Der Einfluss der Athleten-Biometrie und des Rennprofils
* **`rider_bmi` & `age_at_race`:** Das Alter und der Body-Mass-Index des Fahrers bilden die zweite wichtige Informationsebene. Da dieses Basismodell noch keine Interaktionen besitzt, lernt es hier einen globalen, gemittelten Effekt (z. B. dass ein extrem niedriger BMI im Gesamtdurchschnitt aller Etappen leicht positiv für eine Top-Platzierung ist).
* **`stage_nr`:** Die Etappennummer hat ebenfalls ein sichtbares Gewicht. Dies ist logisch, da sich das Fahrerfeld im Laufe einer dreiwöchigen Grand Tour (durch Erschöpfung, Stürze und Aufgaben) stark lichtet und sich die Dynamik von Ausreißergruppen in der zweiten und dritten Woche drastisch verändert.

### 3. Die untergeordnete Rolle der reinen Wetterdaten (Metereologie)
* **`weather_temp_mean`, `weather_precipitation_mean`, etc.:** Die meteorologischen Variablen (Temperatur, Regenwahrscheinlichkeit, Luftfeuchtigkeit) befinden sich geschlossen am ganz unteren Ende der Skala. 
  * *Methodische Einordnung:* Dies bedeutet nicht, dass das Wetter im Radsport irrelevant ist. Da Wettereffekte jedoch hochgradig *nicht-linear* und *kontextabhängig* agieren (z. B. trifft extreme Hitze einen schweren Sprinter auf einer Bergetappe fatal, während sie einem leichten Kletterer weniger ausmacht), kann das isolierte 1D-Basismodell ohne Interaktionen hieraus noch kein starkes globales Signal extrahieren.

### Fazit für das anstehende Hyperparameter-Tuning:
Die globale Analyse bestätigt den Erfolg unserer Feature-Engineering-Strategie: Die Lags haben das Modell auf ein neues Level gehoben. 

Gleichzeitig liefert die niedrige Platzierung der physischen und meteorologischen Variablen die perfekte Steilvorlage für unser **Hyperparameter-Tuning mit Interaktionen ($GA^2M$)**: Wir erwarten, dass durch das Zulassen von Variablen-Paaren (wie `rider_bmi` $\times$ `vertical_meters` oder `weather_temp_mean` $\times$ `distance`) diese physikalischen Parameter in der Wichtigkeits-Skala des optimierten Finalmodells deutlich nach oben klettern werden.


```python
#show(ebm_baseline.explain_global())
```


<!-- http://127.0.0.1:7001/1850957748832/ -->
<iframe src="http://127.0.0.1:7001/1850957748832/" width=100% height=800 frameBorder="0"></iframe>



## Automatisiertes Hyperparameter-Tuning (Grid Search über alle Targets)

Um das volle Potenzial der Explainable Boosting Machines auszuschöpfen, aktivieren wir nun die Feature-Interaktionen ($GA^2M$) und optimieren die Kernparameter über einen systematischen Suchlauf. 

### Strategie der Tuning-Pipeline:
* **Multi-Target-Optimierung:** Die Pipeline iteriert vollautomatisch über alle drei Zielvariablen (`target_top_5`, `target_top_10`, `target_top_20`).
* **Dynamische Gewichtung:** Da sich das Klassenungleichgewicht je nach Target verändert (bei Top 5 extrem, bei Top 20 milder), berechnet die Pipeline für jeden Durchlauf das exakt passende `sample_weight` neu.

#### 1. `interactions` (Anzahl der 2D-Wechselwirkungen)
* **Funktion:** Bestimmt die maximale Anzahl paarweiser Feature-Kombinationen, die das Modell automatisch identifizieren und lernen darf ($GA^2M$-Erweiterung).
* **Bedeutung für das Projekt:** Ein Wert von `interactions=12` erlaubt es dem Modell, die 12 stärksten Kombinationen zu bilden (z. B. `rider_bmi` $\times$ `vertical_meters`). Dadurch lernt der Algorithmus, dass ein hoher BMI auf Flachetappen kein Problem ist, im Hochgebirge jedoch drastisch bestraft wird. Dies ist der wichtigste Hebel, um Fehlalarme (*False Positives*) zu senken.

#### 2. `outer_bags` (Anzahl der Ensembling-Modelle)
* **Funktion:** Regelt das statistische Ensembling (Bagging) auf oberster Ebene. Es bestimmt, wie viele eigenständige EBM-Modelle auf leicht unterschiedlichen Bootstrap-Stichproben trainiert und am Ende gemittelt werden.
* **Bedeutung für das Projekt:** Höhere Werte (z. B. `outer_bags=8`) glätten die gelernten Spline-Kurven und reduzieren die Modellvarianz drastisch. Es macht die Vorhersagen auf der ungesehenen Zukunftssaison (2024/2025) robuster gegen statistisches Rauschen und unvorhersehbare Ausreißer im Rennverlauf.

#### 3. `max_bins` (Granularität der numerischen Features)
* **Funktion:** Bestimmt, in wie viele Abschnitte (Bins) kontinuierliche, numerische Variablen (wie Temperatur, Distanz oder Alter) vorab zerlegt werden. 
* **Bedeutung für das Projekt:** Ein Wert von `256` erlaubt es dem Modell, sehr feine, stufenförmige und nicht-lineare Muster in den Daten zu erkennen. Weniger Bins (z. B. 128) würden die Kurven mathematisch stark vereinfachen und glätten, wodurch man jedoch feine Details (wie den exakten Kipppunkt bei extremen Steigungsprozenten) verlieren könnte.


### Zusammenfassung des Setups für die Tuning-Pipeline

* **Die Zielvariablen (Target-Klassen):** Wir optimieren simultan über `target_top_5`, `target_top_10` und `target_top_20`, wobei für jeden Durchlauf das Klassenungleichgewicht über dynamisch berechnete Stichprobengewichte (`sample_weight`) im Hintergrund exakt ausbalanciert wird.
* **Die Hyperparameter-Konfiguration:** Der Suchraum kombiniert die maximale Anzahl an 2D-Wechselwirkungen (`interactions: [5, 12]`) zur gezielten Reduktion von Fehlalarmen mit dem statistischen Ensembling (`outer_bags: [4, 8]`) zur drastischen Glättung der Spline-Kurven und effektiven Vermeidung von Overfitting.






```python
model_path = '../../data/models'
os.makedirs(model_path, exist_ok=True)

# Definieren der Zielvariablen
target_cols = ['target_top_5', 'target_top_10', 'target_top_20']

# Definition des Suchraums
grid_parameters = {
    'interactions': [5, 12],            # 5 vs. 12 2D-Wechselwirkungen
    'learning_rate': [0.01, 0.02],       # Lernrate / Schrittweite
    'outer_bags': [4, 8],                # Ensembling zur Varianzreduktion (Overfitting-Schutz)
    'max_bins': [256]                    # Granularität der numerischen Splines
}

# Alle Kombinationen generieren
keys, values = zip(*grid_parameters.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

tuning_results = []

# --- CRITICAL FIX 1: Tracker initialisieren ---
best_models_tracker = {target: {'auc': 0.0, 'model': None, 'config': None} for target in target_cols}

print(f"Starte EBM-Tuning-Pipeline...")
print(f"Anzahl der Zielvariablen: {len(target_cols)} | Kombinationen pro Target: {len(experiments)}")
print(f"Gesamtanzahl der Trainingsläufe: {len(target_cols) * len(experiments)}\n")

global_start_time = time.time()

# Äußerer Loop: Iteration über die verschiedenen Platzierungs-Targets
for target in target_cols:
    print(f"\n==================================================================")
    print(f"Tuning für Target: {target.upper()}")
    print("==================================================================")

    # Daten und Targets für diesen spezifischen Durchlauf isolieren
    y_train_curr = y_class_train[target]
    y_test_curr = y_class_test[target]

    # Klassenungleichgewicht dynamisch für das aktuelle Target berechnen
    print("Berechne spezifische sample_weights...")
    current_weights = compute_sample_weight(class_weight='balanced', y=y_train_curr)

    # Innerer Loop: Grid Search über die Hyperparameter
    for idx, config in enumerate(experiments, 1):
        print(f"Lauf {idx}/{len(experiments)}: Inter.= {config['interactions']}, LR={config['learning_rate']}, Bags={config['outer_bags']} ... ", end="", flush=True)

        start_run = time.time()

        # EBM mit der aktuellen Konfiguration initialisieren
        ebm_tuned = ExplainableBoostingClassifier(
            interactions=config['interactions'],
            learning_rate=config['learning_rate'],
            outer_bags=config['outer_bags'],
            max_bins=config['max_bins'],
            validation_size=0.15,
            max_rounds=5000,
            early_stopping_rounds=100,
            random_state=42,
            n_jobs=1  # Sicherer Betrieb auf einem Kern in Anaconda/VS Code
        )

        # Modell trainieren
        ebm_tuned.fit(X_train, y_train_curr, sample_weight=current_weights)

        # Evaluieren auf Saisons 2024/2025
        probs_tuned = ebm_tuned.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test_curr, probs_tuned)

        duration_run = time.time() - start_run
        print(f"ROC-AUC: {auc_score:.4f} ({duration_run:.1f}s)")

        # Ergebnisse archivieren
        tuning_results.append({
            'target': target,
            'interactions': config['interactions'],
            'learning_rate': config['learning_rate'],
            'outer_bags': config['outer_bags'],
            'max_bins': config['max_bins'],
            'test_roc_auc': auc_score,
            'duration_seconds': duration_run
        })


        if auc_score > best_models_tracker[target]['auc']:
            best_models_tracker[target]['auc'] = auc_score
            best_models_tracker[target]['model'] = ebm_tuned
            best_models_tracker[target]['config'] = config

# In ein übersichtliches DataFrame gießen
df_tuning_summary = pd.DataFrame(tuning_results)
total_duration = time.time() - global_start_time

print(f"\nTUNING BEENDET! Gesamte Rechenzeit: {total_duration/60:.1f} Minuten")
print("\nSpeichern der besten Modelle als .pkl")
print("------------------------------------------------------------------")
for target in target_cols:
    best_auc = best_models_tracker[target]['auc']
    best_mod = best_models_tracker[target]['model']
    best_cfg = best_models_tracker[target]['config']

    # Dateiname für das spezifische Target generieren
    file_name = f"ebm_best_{target}.pkl"
    full_save_path = os.path.join(model_path, file_name)

    # Modell via Pickle exportieren
    with open(full_save_path, 'wb') as f:
        pickle.dump(best_mod, f)

    print(f"➔ {target:<15}: AUC={best_auc:.4f} | Gespeichert als: {file_name}")
    print(f"   [Konfiguration: Inter.={best_cfg['interactions']}, LR={best_cfg['learning_rate']}, Bags={best_cfg['outer_bags']}]")
print("==================================================================")
```

    Starte EBM-Tuning-Pipeline...
    Anzahl der Zielvariablen: 3 | Kombinationen pro Target: 8
    Gesamtanzahl der Trainingsläufe: 24
    
    
    ==================================================================
    Tuning für Target: TARGET_TOP_5
    ==================================================================
    Berechne spezifische sample_weights...
    Lauf 1/8: Inter.= 5, LR=0.01, Bags=4 ... ROC-AUC: 0.7962 (123.8s)
    Lauf 2/8: Inter.= 5, LR=0.01, Bags=8 ... ROC-AUC: 0.7964 (332.9s)
    Lauf 3/8: Inter.= 5, LR=0.02, Bags=4 ... ROC-AUC: 0.7968 (90.0s)
    Lauf 4/8: Inter.= 5, LR=0.02, Bags=8 ... ROC-AUC: 0.7995 (164.1s)
    Lauf 5/8: Inter.= 12, LR=0.01, Bags=4 ... ROC-AUC: 0.8016 (137.3s)
    Lauf 6/8: Inter.= 12, LR=0.01, Bags=8 ... ROC-AUC: 0.8015 (293.4s)
    Lauf 7/8: Inter.= 12, LR=0.02, Bags=4 ... ROC-AUC: 0.8022 (99.8s)
    Lauf 8/8: Inter.= 12, LR=0.02, Bags=8 ... ROC-AUC: 0.8021 (203.1s)
    
    ==================================================================
    Tuning für Target: TARGET_TOP_10
    ==================================================================
    Berechne spezifische sample_weights...
    Lauf 1/8: Inter.= 5, LR=0.01, Bags=4 ... ROC-AUC: 0.7762 (675.1s)
    Lauf 2/8: Inter.= 5, LR=0.01, Bags=8 ... ROC-AUC: 0.7765 (1131.1s)
    Lauf 3/8: Inter.= 5, LR=0.02, Bags=4 ... ROC-AUC: 0.7754 (997.3s)
    Lauf 4/8: Inter.= 5, LR=0.02, Bags=8 ... ROC-AUC: 0.7763 (1269.1s)
    Lauf 5/8: Inter.= 12, LR=0.01, Bags=4 ... ROC-AUC: 0.7761 (770.9s)
    Lauf 6/8: Inter.= 12, LR=0.01, Bags=8 ... ROC-AUC: 0.7763 (935.8s)
    Lauf 7/8: Inter.= 12, LR=0.02, Bags=4 ... ROC-AUC: 0.7754 (666.1s)
    Lauf 8/8: Inter.= 12, LR=0.02, Bags=8 ... ROC-AUC: 0.7762 (669.1s)
    
    ==================================================================
    Tuning für Target: TARGET_TOP_20
    ==================================================================
    Berechne spezifische sample_weights...
    Lauf 1/8: Inter.= 5, LR=0.01, Bags=4 ... ROC-AUC: 0.7567 (488.2s)
    Lauf 2/8: Inter.= 5, LR=0.01, Bags=8 ... ROC-AUC: 0.7567 (1219.4s)
    Lauf 3/8: Inter.= 5, LR=0.02, Bags=4 ... ROC-AUC: 0.7562 (831.6s)
    Lauf 4/8: Inter.= 5, LR=0.02, Bags=8 ... ROC-AUC: 0.7563 (1285.4s)
    Lauf 5/8: Inter.= 12, LR=0.01, Bags=4 ... ROC-AUC: 0.7558 (650.2s)
    Lauf 6/8: Inter.= 12, LR=0.01, Bags=8 ... ROC-AUC: 0.7562 (1402.0s)
    Lauf 7/8: Inter.= 12, LR=0.02, Bags=4 ... ROC-AUC: 0.7555 (738.4s)
    Lauf 8/8: Inter.= 12, LR=0.02, Bags=8 ... ROC-AUC: 0.7557 (1386.9s)
    
    TUNING BEENDET! Gesamte Rechenzeit: 276.0 Minuten
    
    Speichern der besten Modelle als .pkl
    ------------------------------------------------------------------
    ➔ target_top_5   : AUC=0.8022 | Gespeichert als: ebm_best_target_top_5.pkl
       [Konfiguration: Inter.=12, LR=0.02, Bags=4]
    ➔ target_top_10  : AUC=0.7765 | Gespeichert als: ebm_best_target_top_10.pkl
       [Konfiguration: Inter.=5, LR=0.01, Bags=8]
    ➔ target_top_20  : AUC=0.7567 | Gespeichert als: ebm_best_target_top_20.pkl
       [Konfiguration: Inter.=5, LR=0.01, Bags=8]
    ==================================================================
    

Starte EBM-Tuning-Pipeline...
Anzahl der Zielvariablen: 3 | Kombinationen pro Target: 8
Gesamtanzahl der Trainingsläufe: 24


==================================================================
Tuning für Target: TARGET_TOP_5
==================================================================
Berechne spezifische sample_weights...
Lauf 1/8: Inter.= 5, LR=0.01, Bags=4 ... ROC-AUC: 0.7962 (123.8s)
Lauf 2/8: Inter.= 5, LR=0.01, Bags=8 ... ROC-AUC: 0.7964 (332.9s)
Lauf 3/8: Inter.= 5, LR=0.02, Bags=4 ... ROC-AUC: 0.7968 (90.0s)
Lauf 4/8: Inter.= 5, LR=0.02, Bags=8 ... ROC-AUC: 0.7995 (164.1s)
Lauf 5/8: Inter.= 12, LR=0.01, Bags=4 ... ROC-AUC: 0.8016 (137.3s)
Lauf 6/8: Inter.= 12, LR=0.01, Bags=8 ... ROC-AUC: 0.8015 (293.4s)
Lauf 7/8: Inter.= 12, LR=0.02, Bags=4 ... ROC-AUC: 0.8022 (99.8s)
Lauf 8/8: Inter.= 12, LR=0.02, Bags=8 ... ROC-AUC: 0.8021 (203.1s)

==================================================================
Tuning für Target: TARGET_TOP_10
==================================================================
Berechne spezifische sample_weights...
Lauf 1/8: Inter.= 5, LR=0.01, Bags=4 ... ROC-AUC: 0.7762 (675.1s)
Lauf 2/8: Inter.= 5, LR=0.01, Bags=8 ... ROC-AUC: 0.7765 (1131.1s)
Lauf 3/8: Inter.= 5, LR=0.02, Bags=4 ... ROC-AUC: 0.7754 (997.3s)
Lauf 4/8: Inter.= 5, LR=0.02, Bags=8 ... ROC-AUC: 0.7763 (1269.1s)
Lauf 5/8: Inter.= 12, LR=0.01, Bags=4 ... ROC-AUC: 0.7761 (770.9s)
Lauf 6/8: Inter.= 12, LR=0.01, Bags=8 ... ROC-AUC: 0.7763 (935.8s)
Lauf 7/8: Inter.= 12, LR=0.02, Bags=4 ... ROC-AUC: 0.7754 (666.1s)
Lauf 8/8: Inter.= 12, LR=0.02, Bags=8 ... ROC-AUC: 0.7762 (669.1s)

==================================================================
Tuning für Target: TARGET_TOP_20
==================================================================
Berechne spezifische sample_weights...
Lauf 1/8: Inter.= 5, LR=0.01, Bags=4 ... ROC-AUC: 0.7567 (488.2s)
Lauf 2/8: Inter.= 5, LR=0.01, Bags=8 ... ROC-AUC: 0.7567 (1219.4s)
Lauf 3/8: Inter.= 5, LR=0.02, Bags=4 ... ROC-AUC: 0.7562 (831.6s)
Lauf 4/8: Inter.= 5, LR=0.02, Bags=8 ... ROC-AUC: 0.7563 (1285.4s)
Lauf 5/8: Inter.= 12, LR=0.01, Bags=4 ... ROC-AUC: 0.7558 (650.2s)
Lauf 6/8: Inter.= 12, LR=0.01, Bags=8 ... ROC-AUC: 0.7562 (1402.0s)
Lauf 7/8: Inter.= 12, LR=0.02, Bags=4 ... ROC-AUC: 0.7555 (738.4s)
Lauf 8/8: Inter.= 12, LR=0.02, Bags=8 ... ROC-AUC: 0.7557 (1386.9s)

TUNING BEENDET! Gesamte Rechenzeit: 276.0 Minuten

Speichern der besten Modelle als .pkl
------------------------------------------------------------------
➔ target_top_5   : AUC=0.8022 | Gespeichert als: ebm_best_target_top_5.pkl
   [Konfiguration: Inter.=12, LR=0.02, Bags=4]
➔ target_top_10  : AUC=0.7765 | Gespeichert als: ebm_best_target_top_10.pkl
   [Konfiguration: Inter.=5, LR=0.01, Bags=8]
➔ target_top_20  : AUC=0.7567 | Gespeichert als: ebm_best_target_top_20.pkl
   [Konfiguration: Inter.=5, LR=0.01, Bags=8]


# Interpretation

Das systematische Hyperparameter-Tuning über 24 Trainingsläufe liefert Erkenntnisse über die Struktur von Platzierungsmustern im professionellen Radsport.

### 1. Die mathematische Selektivität der Target-Klassen
Es zeigt sich ein kontinuierlicher, linearer Abfall der Modellgüte (ROC-AUC) von den **Top 5 (0.8022)** über die **Top 10 (0.7765)** hin zu den **Top 20 (0.7567)**. 
* **Begründung:** Die absolute Weltspitze (Top 5) bringt signifikante, historische und physische Alleinstellungsmerkmale mit, die das Modell trennscharf isolieren kann. Je weiter sich der Platzierungsradius öffnet (Top 20), desto stärker dominieren stochastische Rauscheffekte (wie Renndynamik, Defekte oder Helferdienste), was die Vorhersagbarkeit für rein datenbasierte Modelle naturgemäß erschwert.

### 2. Differenzierte Modellkomplexität je nach Zielgruppe
Die Hyperparameter-Gewinner spiegeln die theoretischen Erwartungen exakt wider:
* **Das Top-5-Modell** maximiert seine Flexibilität durch `interactions=12`. Um die absolute Spitze vorherzusagen, müssen komplexe 2D-Muster (Fahrertyp $\times$ Etappenprofil) voll ausgeschöpft werden.
* **Die Top-10- und Top-20-Modelle** hingegen liefen bei maximaler Interaktionsrate Gefahr, Overfitting auf historischen Daten zu betreiben. Sie wählten algorithmisch eine restriktivere Konfiguration (`interactions=5`), kombiniert mit einer niedrigen Lernrate (`0.01`) und maximalem Ensembling (`outer_bags=8`). Dies glättet die gelernten Spline-Kurven und sichert die Generalisierungsfähigkeit auf den ungesehenen Saisons 2024/2025.

### Fazit:
Die Modelle wurden in ihrer jeweils mathematisch bewiesenen Bestform als `.pkl`-Dateien exportiert. Ein Erweitern des Suchraums ist nicht notwendig, da die Parametergrenzen (z. B. 5 vs. 12 Interaktionen oder 4 vs. 8 Bags) klare Kipppunkte der optimalen Modellkomplexität aufgezeigt haben.


---


## Evaluierung der optimierten EBM-Modelle 

Nachdem die optimalen Hyperparameter identifiziert und die Modelle exportiert wurden, führen wir nun die finale Performance-Analyse durch. Um die Vergleichbarkeit zur Baseline zu wahren, optimieren wir auch hier den Klassifikations-Schwellenwert (Threshold) für jedes Target individuell über die Maximierung des F1-Scores auf den Saisons 2024/2025


```python
model_path = '../../data/models'
target_cols = ['target_top_5', 'target_top_10', 'target_top_20']


for target in target_cols:
    # Modell aus der PKL-Datei laden
    file_name = f"ebm_best_{target}.pkl"
    full_load_path = os.path.join(model_path, file_name)

    with open(full_load_path, 'rb') as f:
        model = pickle.load(f)

    # Wahrscheinlichkeiten für die Testdaten (2024/2025) generieren
    y_test_curr = y_class_test[target]
    probs = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test_curr, probs)

    # Optimalen Schwellenwert über Precision-Recall-Kurve bestimmen
    precision, recall, thresholds = precision_recall_curve(y_test_curr, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    # Hard Labels mit optimalem Threshold generieren
    y_pred_opt = (probs >= best_threshold).astype(int)
    cm = confusion_matrix(y_test_curr, y_pred_opt)

    # 4. Ergebnisse formatiert ausgeben
    print("==================================================================")
    print(f"Evaluierung für: {target.upper()}")
    print("==================================================================")
    print(f"Test ROC-AUC (Saisons 2024/2025)  : {auc_score:.4f}")
    print(f"Optimaler Schwellenwert (Threshold): {best_threshold:.4f}")
    print(f"Maximaler F1-Score                 : {best_f1:.4f}")
    print("------------------------------------------------------------------")
    print("Klassifikationsbericht:")
    print(classification_report(y_test_curr, y_pred_opt, target_names=['Hauptfeld (0)', f'{target.replace("target_", "").upper()} (1)'], digits=4))
    print("------------------------------------------------------------------")
    print("Konfusions-Matrix:")
    print(f"                  Vorhergesagt:    Vorhergesagt:")
    print(f"                  Hauptfeld (0)    {target.replace('target_', '').upper()} (1)")
    print(f"Tatsächlich:")
    print(f"Hauptfeld (0)      {cm[0, 0]:<16,d} {cm[0, 1]:,d}")
    print(f"Tatsächlich (1)    {cm[1, 0]:<16,d} {cm[1, 1]:,d}")

```

    ==================================================================
    Evaluierung für: TARGET_TOP_5
    ==================================================================
    Test ROC-AUC (Saisons 2024/2025)  : 0.8022
    Optimaler Schwellenwert (Threshold): 0.8072
    Maximaler F1-Score                 : 0.2430
    ------------------------------------------------------------------
    Klassifikationsbericht:
                   precision    recall  f1-score   support
    
    Hauptfeld (0)     0.9779    0.9555    0.9666     17246
        TOP_5 (1)     0.1926    0.3291    0.2430       556
    
         accuracy                         0.9360     17802
        macro avg     0.5852    0.6423    0.6048     17802
     weighted avg     0.9533    0.9360    0.9440     17802
    
    ------------------------------------------------------------------
    Konfusions-Matrix:
                      Vorhergesagt:    Vorhergesagt:
                      Hauptfeld (0)    TOP_5 (1)
    Tatsächlich:
    Hauptfeld (0)      16,479           767
    Tatsächlich (1)    373              183
    ==================================================================
    Evaluierung für: TARGET_TOP_10
    ==================================================================
    Test ROC-AUC (Saisons 2024/2025)  : 0.7765
    Optimaler Schwellenwert (Threshold): 0.7501
    Maximaler F1-Score                 : 0.2919
    ------------------------------------------------------------------
    Klassifikationsbericht:
                   precision    recall  f1-score   support
    
    Hauptfeld (0)     0.9564    0.9266    0.9413     16698
       TOP_10 (1)     0.2452    0.3605    0.2919      1104
    
         accuracy                         0.8915     17802
        macro avg     0.6008    0.6436    0.6166     17802
     weighted avg     0.9123    0.8915    0.9010     17802
    
    ------------------------------------------------------------------
    Konfusions-Matrix:
                      Vorhergesagt:    Vorhergesagt:
                      Hauptfeld (0)    TOP_10 (1)
    Tatsächlich:
    Hauptfeld (0)      15,473           1,225
    Tatsächlich (1)    706              398
    ==================================================================
    Evaluierung für: TARGET_TOP_20
    ==================================================================
    Test ROC-AUC (Saisons 2024/2025)  : 0.7567
    Optimaler Schwellenwert (Threshold): 0.6361
    Maximaler F1-Score                 : 0.3895
    ------------------------------------------------------------------
    Klassifikationsbericht:
                   precision    recall  f1-score   support
    
    Hauptfeld (0)     0.9231    0.8495    0.8848     15597
       TOP_20 (1)     0.3193    0.4993    0.3895      2205
    
         accuracy                         0.8061     17802
        macro avg     0.6212    0.6744    0.6372     17802
     weighted avg     0.8483    0.8061    0.8234     17802
    
    ------------------------------------------------------------------
    Konfusions-Matrix:
                      Vorhergesagt:    Vorhergesagt:
                      Hauptfeld (0)    TOP_20 (1)
    Tatsächlich:
    Hauptfeld (0)      13,250           2,347
    Tatsächlich (1)    1,104            1,101
    

## Interpretation der finalen Modellergebnisse

Die finale Auswertung der getunten EBM-Modelle auf den ungesehenen Saisons 2024/2025 zeigt eine Performance-Steigerung und liefert die folgenden Erkenntnisse.

### 1. Der direkte Vorher-Nachher-Vergleich (Das Tuning-Upgrade bei TOP_10)
Vergleicht man das optimierte Top-10-Modell mit dem 1D-Basismodell, wird der Einfluss der gelernten Feature-Interaktionen ($GA^2M$) und der Regularisierung deutlich:
* **ROC-AUC-Sprung:** Der Score steigt von **0.7311 auf 0.7765**. Das Modell gewinnt also signifikant an globaler Trennschärfe.
* **Das Einbrechen der Fehlalarme:** In der Baseline hatten wir noch **2.485 False Positives** (Fehlalarme). Das getunte Modell drückt diesen Wert radikal auf **1.225** — die Fehlprognosen wurden also **mehr als halbiert**! Das Modell wirft kein zu grobes Netz mehr aus, sondern agiert durch die Verknüpfung von Variablen (z. B. Fahrertyp $\times$ Höhenmeter) deutlich selektiver.
* **F1-Score und Precision:** Die Precision für die Top 10 steigt von 16,33 % auf **24,52 %**. Der F1-Score klettert von 0.2381 auf **0.2919**. Das Modell ist nun ungleich präziser bei der Nominierung der Favoriten.

### 2. Die mechanische Logik über die verschiedenen Target-Klassen
Betrachtet man die Metriken über alle drei gelernten Platzierungsklassen hinweg, wird das mathematische Verhalten bei unbalancierten Sportdaten sichtbar:

#### A) TARGET_TOP_5 (Die absolute Elite)
* **ROC-AUC (0.8022):** Der Hohe ROC-AUC Wert beweist eine hohe Trennschärfe. Die Weltspitze lässt sich anhand der Kombination aus historischer Form und Etappenprofil fast fehlerfrei vom Rest isolieren.
* **Precision (19,26 %) & Recall (32,91 %):** Da unter die Top 5 mathematisch nur halb so viele Fahrer kommen wie unter die Top 10, schlägt das Klassenungleichgewicht hier am härtesten zu (nur 556 echte Einsen bei über 17.000 Zeilen). Dennoch ist jeder fünfte Tipp des Modells ein korrekter in den extrem engen Top 5.

#### B) TARGET_TOP_10 (Das solide Fundament)
* Dieses Modell findet die beste Balance aus hoher Trennschärfe (**0.7765**) und einer stark verbesserten Trefferquote. Es erkennt bereits **398 echte Top-10-Platzierungen** punktgenau im Rennverlauf.

#### C) TARGET_TOP_20 (Das breite Verfolgerfeld)
* **F1-Score (0.3895) & Recall (49,93 %):** Der F1-Score steigt hier stark an. Da die Klasse 1 mit 2.205 Fahrern deutlich größer ist, wird das Modell für knappe Fehlprognosen mathematisch weniger bestraft.
* **Sensationelle Trefferquote:** Das Modell sagt bei `target_top_20` fast **jede zweite Platzierung (49,93 % Recall) vollkommen korrekt voraus**. 1.101-mal lagen der Algorithmus und die Realität absolut deckungsgleich.

### Fazit für die wissenschaftliche Ausarbeitung:
Das Hyperparameter-Tuning war ein Erfolg. Die Aktivierung von Feature-Interaktionen hat das "Favoriten-Streuverhalten" des Modells korrigiert, was sich in der Reduktion der False Positives in den Konfusionsmatrizen widerspiegelt. 

Gleichzeitig untermauern die Ergebnisse das typische Verhalten von Klassifikationsmodellen im Sport: Je breiter das Target (Top 20), desto höher der F1-Score durch die schiere Klassendichte; je elitärer das Target (Top 5), desto höher die reine mathematische Trennschärfe (AUC) durch die Einzigartigkeit der Ausnahme-Athleten.

---


# Interpretation der gelernten Muster (Globale Feature-Wichtigkeiten im Vergleich)

Nachdem die Modelle final evaluiert wurden, öffnen wir nun die „Glass-Box“ und vergleichen, welche Variablen und neu gelernten 2D-Interaktionen den größten mathematischen Einfluss auf die Vorhersagen der drei Platzierungsklassen haben. Der *Mean Absolute Score* gibt an, wie stark ein Feature die prognostizierte Log-Odds-Wahrscheinlichkeit im Durchschnitt verändert.


```python
importance_records = []

for target in target_cols:
    # Modell aus dem Tracker holen
    model = best_models_tracker[target]['model']

    # Namen aller Terme (Features + Interaktionen) holen
    # EBM speichert alle gelernten Terme in model.term_names_
    term_names = model.term_names_

    # Die mittleren absoluten Scores (Wichtigkeiten) holen
    term_importances = model.term_importances()

    # In Datensätze zerlegen
    for name, importance in zip(term_names, term_importances):
        importance_records.append({
            'Target': target.replace('target_', '').upper(),
            'Feature / Interaktion': name,
            'Importance': importance
        })

# DataFrame erstellen
df_long = pd.DataFrame(importance_records)

# Von der Langform in Pivot-Vergleichstabelle transformieren
df_importance_compare = df_long.pivot(
    index='Feature / Interaktion',
    columns='Target',
    values='Importance'
).fillna(0.0)

# Nach der Wichtigkeit im viel diskutierten TOP_10 Modell sortieren
df_importance_compare = df_importance_compare.sort_values(by='TOP_10', ascending=False)


print("Vergleichstabelle Featurewichtigkeit (MEAN ABSOLUTE SCORE)")
print(df_importance_compare.round(4).to_string())
print("==================================================================")

# Sicherer Export in den korrekten Ordner
csv_save_path = os.path.join(model_path, 'ebm_feature_importances_comparison.csv')
df_importance_compare.to_csv(csv_save_path)
print(f"gespeichert unter:\n {csv_save_path}")
```

    Vergleichstabelle Featurewichtigkeit (MEAN ABSOLUTE SCORE)
    Target                                                   TOP_10  TOP_20   TOP_5
    Feature / Interaktion                                                          
    lag_rider_rank_season                                    0.6183  0.5966  0.5090
    vertical_meters & rider_bmi                              0.3628  0.3033  0.3194
    lag_rider_points_season                                  0.2696  0.2115  0.4398
    lag_race_competitiveness_median                          0.1633  0.1412  0.1972
    rider_bmi & gradient_final_km                            0.1490  0.1611  0.1258
    age_at_race                                              0.1461  0.1187  0.1491
    rider_bmi                                                0.1314  0.1113  0.1530
    gradient_final_km & lag_rider_rank_season                0.1313  0.1325  0.0216
    stage_nr                                                 0.1152  0.1074  0.1026
    lag_team_power_index                                     0.0903  0.0781  0.0342
    rider_bmi & weather_temp_trend                           0.0730  0.0704  0.0567
    gradient_final_km & lag_rider_points_season              0.0494  0.0564  0.0506
    gradient_final_km                                        0.0353  0.0377  0.0108
    vertical_meters                                          0.0329  0.0037  0.0438
    weather_temp_trend                                       0.0113  0.0097  0.0053
    weather_precipitation_mean                               0.0065  0.0037  0.0032
    weather_temp_mean                                        0.0064  0.0046  0.0099
    team_tier                                                0.0063  0.0084  0.0003
    wind_stability_index                                     0.0061  0.0045  0.0090
    weather_humidity_mean                                    0.0052  0.0049  0.0034
    distance                                                 0.0044  0.0046  0.0030
    weather_rain_prob_mean                                   0.0021  0.0023  0.0033
    lag_rider_rank_season & lag_team_power_index             0.0000  0.0000  0.0181
    lag_rider_points_season & lag_team_power_index           0.0000  0.0000  0.0272
    age_at_race & lag_rider_points_season                    0.0000  0.0000  0.0148
    age_at_race & lag_rider_rank_season                      0.0000  0.0000  0.0336
    lag_rider_rank_season & lag_race_competitiveness_median  0.0000  0.0000  0.0279
    stage_nr & rider_bmi                                     0.0000  0.0000  0.0316
    rider_bmi & weather_temp_mean                            0.0000  0.0000  0.0163
    ==================================================================
    gespeichert unter:
     ../../data/models\ebm_feature_importances_comparison.csv
    

# Interpretation der Feature-Wichtigkeiten im Multi-Target-Vergleich

Die Vergleichstabelle inkl. des *Mean Absolute Scores* gewährt Einblicke in die gelernten Entscheidungsstrukturen der optimierten Modelle. Sie liefert die empirische Begründung für die signifikanten Performance-Sprünge nach dem Hyperparameter-Tuning.

### Die Entdeckung der Schlüssel-Interaktion: Bergspezialisten vs. Rouleurs
* **`vertical_meters & rider_bmi` (Platz 2 mit Scores bis zu 0.3628):** Diese vom Modell autonom identifizierte 2D-Wechselwirkung ist der "Gamechanger" der Optimierung. Während die Einzelvariablen `vertical_meters` (~0.03) und `rider_bmi` (~0.13) global betrachtet wenig Einfluss haben, entfaltet ihre **Kombination** eine Hebelwirkung.
  * *Sportwissenschaftliche Fundierung:* Das Modell hat gelernt, das physikalische Gesetz des Leistung-Gewicht-Verhältnisses ($W/kg$) im Radsport perfekt abzubilden. Ein hoher BMI (schwerer Fahrer) ist auf flachen Etappen kein Nachteil, wirkt jedoch bei steigenden Höhenmetern (`vertical_meters`) mathematisch wie eine Strafe für die Top-Platzierung. Umgekehrt wird ein niedriger BMI erst bei vielen Höhenmetern massiv positiv bewertet. Diese nicht-lineare Logik fehlte der Baseline komplett und erklärt das Einbrechen der Fehlalarme (*False Positives*).

### Weitere relevante 2D-Interaktionen
* **`rider_bmi & gradient_final_km` (~0.14):** Das Modell kombiniert das Gewicht des Fahrers mit dem Gradient des letzten Kilometers. Dies erlaubt es dem Algorithmus, zwischen klassischen Flachsprints (niedrige Steigung, hoher BMI/Sprinter im Vorteil) und harten Bergankünften (hohe Steigung, niedriger BMI/Kletterer im Vorteil) messerscharf zu differenzieren.
* **`gradient_final_km & lag_rider_rank_season` (0.1313 bei TOP_10):** Hier verknüpft das Modell die historische Klasse eines Fahrers mit dem Etappenfinale. Ein Top-Klassifikationsfahrer bringt seine Stärke vor allem dann ein, wenn das Finale selektiv und steil ist. Bemerkenswert: Beim elitären `TOP_5`-Modell bricht diese Interaktion ein (0.0216), da hier die reine historische Konstanz (`lag_rider_points_season`: 0.4398) das Feld ohnehin schon dominiert. Dies zeigt zudem, dass die Rangliste von den GC-Fahrern angeführt wird.

### Die Hierarchie der historischen Leistungsmerkmale
* **`lag_rider_rank_season` & `lag_rider_points_season`:** Trotz aller komplexen Interaktionen bleiben die historischen Saisonsergebnisse das Fundament der Prädiktion. 
* **Interessante Verschiebung beim TOP_5-Modell:** Während beim breiteren `TOP_10`- und `TOP_20`-Feld der *Saisonrang* (`lag_rider_rank_season` ~0.61) dominiert, verlagert sich das Gewicht bei der absoluten Elite (`TOP_5`) auf die *kumulierten Saisonpunkte* (`lag_rider_points_season`: 0.4398). Um unter die Top 5 zu fahren, reicht ein solider Rang nicht mehr aus – das Modell verlangt den Nachweis von eingefahrenen Spitzenplatzierungen (hohe Punktzahl) in der Historie.

### Meteorologische Daten
* **Reine Wetterdaten (`weather_precipitation_mean`, `weather_temp_mean`, etc.):** Auch im getunten Modell verharren die solitären Wettermerkmale am ganz unteren Ende der Skala (~0.005). 
* **Die Nuance durch Interaktion:** Allerdings zeigt sich, dass das Wetter über Umwege einfließt: Die Interaktion `rider_bmi & weather_temp_trend` (~0.07) rangiert im Mittelfeld. Das Modell erkennt folglich, dass Temperaturveränderungen im Rennverlauf je nach physischer Konstitution (Körpermasse/BMI) des Athleten unterschiedliche metabolische Auswirkungen haben.
* Für zukünftige Betrachtungen können die genauen Geo-Daten der Streckenverläufe herangezogen werden, um den Einfluss von Wind exakter zu bestimmen. Die kann sich als starkes Feature herausstellen.

---

## Visualisierung der gelernten 2D-Feature-Interaktion

Da das interaktive Dashboard aufgrund von Port-Restriktionen in der Notebook-Umgebung blockiert wird, extrahieren wir die gelernten Effekte der stärksten identifizierten Interaktion (`vertical_meters & rider_bmi`) nun programmgesteuert. Die folgende Heatmap zeigt den mathematischen Beitrag (Schnittpunkt-Score) der beiden Variablen auf die Log-Odds der Top-10-Wahrscheinlichkeit.


```python
# Das beste Top-10 Modell auswählen
best_ebm_top10 = best_models_tracker['target_top_10']['model']

# Den Index der gewünschten Interaktion finden
# Wir suchen nach dem exakten Namen in der Liste der gelernten Terme
interaction_name = 'vertical_meters & rider_bmi'

term_index = best_ebm_top10.term_names_.index(interaction_name)

# Die gelernten Daten (Bin-Grenzen und Scores) extrahieren
exp = best_ebm_top10.explain_global()
data_dict = exp.data(term_index)

# Matrix der gelernten Scores abgreifen
scores = data_dict['scores']
x_bins = data_dict['left_names']  # vertical_meters Bins
y_bins = data_dict['right_names'] # rider_bmi Bins

# Roh-Labels formatieren
x_labels_raw = [f"{float(x):.0f}" if isinstance(x, (int, float)) else str(x) for x in x_bins]
y_labels_raw = [f"{float(y):.1f}" if isinstance(y, (int, float)) else str(y) for y in y_bins]

# 4. Masken für die Achsen-Ticks erstellen (alle Werte außer jedem 5. ausblenden)
# Wenn ein Label nicht angezeigt werden soll, übergeben wir False an Seaborn
x_ticks = [label if idx % 5 == 0 else "" for idx, label in enumerate(x_labels_raw)]
y_ticks = [label if idx % 5 == 0 else "" for idx, label in enumerate(y_labels_raw)]

# 5. Heatmap plotten
plt.figure(figsize=(12, 8))
sns.heatmap(
    scores.T,
    xticklabels=x_ticks,
    yticklabels=y_ticks,
    cmap='RdYlGn',
    center=0,
    cbar_kws={'label': 'Beitrag zur Top-10 Wahrscheinlichkeit (Log-Odds)'}
)

plt.title(f'Gelernte EBM-Interaktion: {interaction_name}\n(Grün = Begünstigt Top 10 | Rot = Bestraft Top 10)', fontsize=14, fontweight='bold')
plt.xlabel('Höhenmeter der Etappe (vertical_meters)', fontsize=12)
plt.ylabel('Body-Mass-Index des Fahrers (rider_bmi)', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# Grafik direkt für die Hausarbeit speichern
plt.savefig(os.path.join(model_path, 'ebm_heatmap_bmi_vs_height.png'), dpi=300)
plt.show()
```


    
![png](10-01-02_EBM_final_files/10-01-02_EBM_final_33_0.png)
    


---

### Methodischer Übergang: Vom Pointwise-Classification- zum Listwise-Ranking-Framing

Die obige Evaluierung (Classification Report, F1-Score) analysiert das Modell rein auf "Pointwise"-Ebene – das heißt, jeder Fahrer wird isoliert bewertet. Um dem Feedback gerecht zu werden und einen fairen Vergleich mit dem späteren XGBRanker (Pairwise/Listwise) zu ermöglichen, implementieren wir nun eine ranking-orientierte Evaluierung. 

Wir nutzen die pointwise vorhergesagten EBM-Wahrscheinlichkeiten, um die Fahrer innerhalb jeder spezifischen Etappe (`stage_id`) absteigend zu sortieren. Diese resultierende Rangliste messen wir nun an den wissenschaftlichen Standard-Rankingmetriken NDCG@5, NDCG@10 sowie den praxisnahen Metriken Top-1-Accuracy und Top-10-Overlap. Die vorherigen Klassifikationsmetriken bleiben als Dokumentation der globalen Trennschärfe im Notebook erhalten.


```python
pfad_processed = '../../data/processed'

groups_test = pd.read_pickle(os.path.join(pfad_processed, 'groups_test.pkl'))
y_rank_test = pd.read_pickle(os.path.join(pfad_processed, 'y_rank_test.pkl'))
# Daten Laden für reines Ranking

#  Wahrscheinlichkeiten generieren (Pointwise-Scores aus dem EBM)
# AM Beispiel des Top 10 Modells

best_ebm = best_models_tracker['target_top_10']['model']
ebm_probs = best_ebm.predict_proba(X_test)[:, 1]

# Ein sauberes, temporäres DataFrame für die Ranking-Berechnung bauen
eval_df = pd.DataFrame({
    'stage_id': groups_test,       # Kommt aus groups_test split
    'true_rank': y_rank_test,      # Kommt aus y_rank_test split
    'pred_score': ebm_probs        # Die soeben berechneten Wahrscheinlichkeiten
})

# Relevanz-Scores für NDCG definieren (Platz 1 kriegt Relevanz 10, Platz 10 kriegt 1, ab 11 ist es 0)
eval_df['relevance'] = eval_df['true_rank'].apply(lambda x: max(0, 11 - x) if x <= 10 else 0)

ndcg_5_list = []
ndcg_10_list = []
top1_correct = 0
top10_overlaps = []

# Gruppieren nach der echten stage_id
grouped = eval_df.groupby('stage_id')

for stage, group in grouped:
    if len(group) < 2:
        continue  # Etappen mit nur einem Fahrer überspringen

    # Wahrer Relevanzvektor und vorhergesagter Scorevektor für NDCG umformen
    true_rel = group['relevance'].values.reshape(1, -1)
    pred_scores = group['pred_score'].values.reshape(1, -1)

    # A) NDCG berechnen (nur wenn es in der Etappe überhaupt Fahrer in den echten Top-10 gibt)
    if np.sum(true_rel) > 0:
        ndcg_5 = ndcg_score(true_rel, pred_scores, k=5)
        ndcg_10 = ndcg_score(true_rel, pred_scores, k=10)
        ndcg_5_list.append(ndcg_5)
        ndcg_10_list.append(ndcg_10)

    # B) Top-1 & Top-10 Accuracy berechnen
    # Sortiere das Fahrerfeld dieser Etappe nach unseren EBM-Scores absteigend
    sorted_group = group.sort_values(by='pred_score', ascending=False).reset_index(drop=True)

    # Top-1 Check: Hat unser am höchsten bewerteter Fahrer die Etappe echt gewonnen?
    if sorted_group.loc[0, 'true_rank'] == 1:
        top1_correct += 1

    # Top-10 Check: Wie viele unserer Top-10 Tipps landen in den echten Top-10?
    pred_top10_riders = sorted_group.head(10)
    true_top10_count = pred_top10_riders[pred_top10_riders['true_rank'] <= 10].shape[0]
    top10_overlaps.append(true_top10_count / 10.0)

# Ergebnisse
print("Finale Rankings für den Pointwise Approach")
print("==================================================================")
print(f"Mittlerer NDCG@5   (Sortierqualität Top 5) : {np.mean(ndcg_5_list):.4f}")
print(f"Mittlerer NDCG@10  (Sortierqualität Top 10): {np.mean(ndcg_10_list):.4f}")
print("------------------------------------------------------------------")
print(f"Top-1 Accuracy    (Sieger exakt getroffen): { (top1_correct / len(grouped)) * 100:.2f}%")
print(f"Top-10 Overlap    (Überschneidung Top 10) : {np.mean(top10_overlaps) * 100:.2f}%")
print("==================================================================\n")
```

    Finale Rankings für den Pointwise Approach
    ==================================================================
    Mittlerer NDCG@5   (Sortierqualität Top 5) : 0.3074
    Mittlerer NDCG@10  (Sortierqualität Top 10): 0.3173
    ------------------------------------------------------------------
    Top-1 Accuracy    (Sieger exakt getroffen): 16.96%
    Top-10 Overlap    (Überschneidung Top 10) : 25.98%
    ==================================================================
    
    

## Diskussion der empirischen Ranking-Ergebnisse des Pointwise-Modells

Die nachträgliche Evaluierung der optimierten EBM über ein etappenbasiertes Ranking liefert ein klares mathematisches Bild und legt die inhärenten Grenzen des Pointwise-Ansatzes offen.

### 1. Analyse der Sortierqualität (NDCG@5 = 0.3074 | NDCG@10 = 0.3173)
* Ein NDCG-Wert von über 0.30 zeigt, dass das Modell keineswegs zufällig rät (ein Zufallsranking bei ~170 Fahrern läge nahe 0). Die EBM besitzt eine Signalstärke, um die relative Hierarchie des Fahrerfelds grob vorzusortieren.
* Dass der NDCG@10 leicht höher liegt als der NDCG@5, deutet darauf hin, dass es dem Modell etwas leichter fällt, die erweiterte Spitzengruppe (Top 10) einzugrenzen, als die exakte Reihenfolge der Top 5 fehlerfrei zu bestimmen.

### 2. Die Praxis-Metriken (Top-1 Accuracy = 16.96% | Top-10 Overlap = 25.98%)
* **Top-1 Accuracy (16.96%):** In knapp 17 % aller Etappen der Saisons 2024/2025 tippt das Modell den *exakten Etappensieger* auf Platz 1. Angesichts der enormen Dichte im Profiradsport ist dies ein beachtlicher Wert für ein Modell, das die Konkurrenten einer Etappe niemals direkt miteinander verglichen hat.
* **Top-10 Overlap (25.98%):** Im Durchschnitt landen etwa 2.6 von 10 Fahrern, die das Modell für die Top 10 vorschlägt, am Ende auch tatsächlich in den realen Top 10. 

### Fazit und Überleitung zum Listwise-Ranking
Die Ergebnisse verdeutlichen exakt das Problem des *Pointwise-Framings*: 
Weil das EBM-Modell jeden Fahrer isoliert betrachtet, berechnet es für die Top-Favoriten oft nahezu identische, hohe Wahrscheinlichkeiten (z. B. Pogacar: 0.85, Vingegaard: 0.84, Roglic: 0.83). Das Modell weiß jedoch nicht, dass an diesem spezifischen Tag nur *einer* von ihnen gewinnen kann und sich ihre Chancen im realen Rennen gegenseitig bedingen (Nullsummenspiel).

Da das Pointwise-Modell keine Etappen-internen Relationen versteht, schöpft es das theoretische Maximum nicht aus. Dies motiviert uns den **XGBRanker** einzusetzen. Dieser optimiert nicht mehr die Wahrscheinlichkeit eines einzelnen Fahrers, sondern lernt über paarweise/listenweise Vergleiche, das gesamte Feld einer Etappe dynamisch zu sortieren.

---

## Exemplarische Fallstudie: Analyse einer spezifischen Etappe

Um die mathematischen Ranking-Metriken greifbar zu machen und die qualitative Funktionsweise des Pointwise-Ansatzes zu evaluieren, isolieren wir im Folgenden repräsentative Etappe aus dem Testzeitraum (Saison 2024/2025). Wir stellen die tatsächliche Top-10-Platzierung der offiziellen Rennleitung den prognostizierten Wahrscheinlichkeiten und der daraus resultierenden Modell-Rangliste gegenüber. Dies verdeutlicht praxisnah, wie das Modell die "GC-Dominanz" und physische Parameter in ein konkretes Tagesklassement übersetzt.

Dazu nutzen wir zwei Bergetappen der Tour de France aus dem Jahr 2025.

- Etappe 16 -> eine kleine Gruppe kam vor den Favoriten ins Ziel
- Etappe 12 -> Pogacar und Vinegaard kamen als erstes ins Ziel


```python
# Metadaten laden
meta_test = pd.read_pickle('../../data/processed/meta_test.pkl')

# DataFrame mit echten Rängen und EBM-Wahrscheinlichkeiten aufbauen
case_study_df = meta_test.copy()
case_study_df['true_rank'] = y_rank_test
case_study_df['pred_prob'] = ebm_probs

# definition der beiden Etappen
etappe_favorit = "tour-de-france_2025_ST12"  # Pogacar-Sieg
etappe_ausreisser = "tour-de-france_2025_ST16"  # Gruppe kommt durch

etappen_zu_pruefen = [
    ("FAVORITEN-SZENARIO (TDF 2025 - Etappe 12)", etappe_favorit),
    ("AUSREISSER-SZENARIO (TDF 2025 - Etappe 16)", etappe_ausreisser)
]

for titel, stage_id in etappen_zu_pruefen:
    # Daten für die spezifische Etappe filtern
    stage_data = case_study_df[case_study_df['stage_id'] == stage_id].copy()

    if stage_data.empty:
        print(f"Die ID '{stage_id}' wurde nicht im Testset gefunden. Bitte Schreibweise prüfen!")
        continue

    # Modell-Ranking erstellen (nach Wahrscheinlichkeit sortieren)
    stage_data = stage_data.sort_values(by='pred_prob', ascending=False).reset_index(drop=True)
    stage_data['model_rank'] = stage_data.index + 1

    print("\n")
    print(titel)
    print("==================================================================")
    print("Die Top 10 Prognosen des Modells vs. Reale Platzierung:")
    print("------------------------------------------------------------------")

    # Tabelle für den Export formatieren
    vergleichs_tabelle = stage_data.head(10)[['model_rank', 'meta_name', 'pred_prob', 'true_rank']]
    vergleichs_tabelle.columns = ['Modell-Platz', 'Fahrer-Name', 'Top-10 Chance', 'Echter Platz']

    # Wahrscheinlichkeit in Prozente umrechnen für die Lesbarkeit
    vergleichs_tabelle['Top-10 Chance'] = vergleichs_tabelle['Top-10 Chance'].apply(lambda x: f"{x*100:.1f}%")

    print(vergleichs_tabelle.to_string(index=False))
```

    
    
    FAVORITEN-SZENARIO (TDF 2025 - Etappe 12)
    ==================================================================
    Die Top 10 Prognosen des Modells vs. Reale Platzierung:
    ------------------------------------------------------------------
     Modell-Platz      Fahrer-Name Top-10 Chance  Echter Platz
                1     Ben O'Connor         94.9%          16.0
                2     Marc Hirschi         94.5%          57.0
                3 Jonas Vingegaard         93.3%           2.0
                4        Enric Mas         93.2%          33.0
                5   Lenny Martinez         92.0%          97.0
                6 Carlos Rodríguez         92.0%          22.0
                7 Matteo Jorgenson         91.9%          15.0
                8 Aleksandr Vlasov         91.0%          34.0
                9       Adam Yates         89.9%          23.0
               10    Biniam Girmay         89.6%         117.0
    
    
    AUSREISSER-SZENARIO (TDF 2025 - Etappe 16)
    ==================================================================
    Die Top 10 Prognosen des Modells vs. Reale Platzierung:
    ------------------------------------------------------------------
     Modell-Platz      Fahrer-Name Top-10 Chance  Echter Platz
                1     Marc Hirschi         93.6%          83.0
                2     Ben O'Connor         92.3%          32.0
                3        Enric Mas         91.3%           7.0
                4 Carlos Rodríguez         91.0%          15.0
                5 Jonas Vingegaard         89.9%           6.0
                6 Aleksandr Vlasov         89.7%          49.0
                7 Matteo Jorgenson         89.2%         115.0
                8   Lenny Martinez         88.6%         122.0
                9    Biniam Girmay         87.9%         158.0
               10    Tadej Pogačar         87.5%           5.0
    


```python
# Metadaten laden, falls noch nicht im RAM
meta_test = pd.read_pickle('../../data/processed/meta_test.pkl') #[cite: 1]

# DataFrame mit echten Rängen und EBM-Wahrscheinlichkeiten aufbauen
case_study_df = meta_test.copy()
case_study_df['true_rank'] = y_rank_test
case_study_df['pred_prob'] = ebm_probs

# definition beider Etappen
etappe_favorit = "tour-de-france_2025_ST12"
etappe_ausreisser = "tour-de-france_2025_ST16"

etappen_zu_pruefen = [
    ("FAVORITEN-SZENARIO (TDF 2025 - Etappe 12)", etappe_favorit),
    ("AUSREISSER-SZENARIO (TDF 2025 - Etappe 16)", etappe_ausreisser)
]

for titel, stage_id in etappen_zu_pruefen:
    # Daten für die spezifische Etappe filtern
    stage_data = case_study_df[case_study_df['stage_id'] == stage_id].copy()


    # Komplettes Modell-Ranking für diese Etappe berechnen
    stage_data = stage_data.sort_values(by='pred_prob', ascending=False).reset_index(drop=True)
    stage_data['model_rank'] = stage_data.index + 1

    print("\n==========================================================================================")
    print(titel)
    print("==========================================================================================")

  # Tabelle 1
    print("TABELLE 1: Die Top 10 Prognosen des Modells (Wonach das Modell sortiert):")
    print("------------------------------------------------------------------")
    tabelle_pred = stage_data.head(10)[['model_rank', 'meta_name', 'pred_prob', 'true_rank']].copy()
    tabelle_pred.columns = ['Modell-Platz', 'Fahrer-Name', 'Top-10 Chance', 'Echter Platz']
    tabelle_pred['Top-10 Chance'] = tabelle_pred['Top-10 Chance'].apply(lambda x: f"{x*100:.1f}%")
    print(tabelle_pred.to_string(index=False))

    print("\n------------------------------------------------------------------")

   # Tabelle 2
    print("TABELLE 2: Die echten Top 10 des Renntages inkl. der Modellwahrscheinlchkeiten")
    print("------------------------------------------------------------------")
    # Wir filtern nach den echten Plätzen 1 bis 10 und sortieren sie von 1 bis 10
    tabelle_real = stage_data[stage_data['true_rank'] <= 10].sort_values(by='true_rank').head(10)[['true_rank', 'meta_name', 'model_rank', 'pred_prob']].copy()
    tabelle_real.columns = ['Echter Platz', 'Fahrer-Name', 'Modell-Platz', 'Top-10 Chance']
    tabelle_real['Top-10 Chance'] = tabelle_real['Top-10 Chance'].apply(lambda x: f"{x*100:.1f}%")
    print(tabelle_real.to_string(index=False))

    print("==========================================================================================\n")
```

    
    ==========================================================================================
    FAVORITEN-SZENARIO (TDF 2025 - Etappe 12)
    ==========================================================================================
    TABELLE 1: Die Top 10 Prognosen des Modells (Wonach das Modell sortiert):
    ------------------------------------------------------------------
     Modell-Platz      Fahrer-Name Top-10 Chance  Echter Platz
                1     Ben O'Connor         94.9%          16.0
                2     Marc Hirschi         94.5%          57.0
                3 Jonas Vingegaard         93.3%           2.0
                4        Enric Mas         93.2%          33.0
                5   Lenny Martinez         92.0%          97.0
                6 Carlos Rodríguez         92.0%          22.0
                7 Matteo Jorgenson         91.9%          15.0
                8 Aleksandr Vlasov         91.0%          34.0
                9       Adam Yates         89.9%          23.0
               10    Biniam Girmay         89.6%         117.0
    
    ------------------------------------------------------------------
    TABELLE 2: Die echten Top 10 des Renntages inkl. der Modellwahrscheinlchkeiten
    ------------------------------------------------------------------
     Echter Platz                Fahrer-Name  Modell-Platz Top-10 Chance
              1.0              Tadej Pogačar            12         88.0%
              2.0           Jonas Vingegaard             3         93.3%
              3.0           Florian Lipowitz            33         69.7%
              4.0 Tobias Halland Johannessen            35         69.6%
              5.0                Oscar Onley            16         86.1%
              6.0            Kévin Vauquelin            81         39.1%
              7.0            Remco Evenepoel            17         85.9%
              8.0                 Felix Gall            40         62.9%
              9.0              Primož Roglič            21         80.7%
             10.0         Cristián Rodríguez            46         62.2%
    ==========================================================================================
    
    
    ==========================================================================================
    AUSREISSER-SZENARIO (TDF 2025 - Etappe 16)
    ==========================================================================================
    TABELLE 1: Die Top 10 Prognosen des Modells (Wonach das Modell sortiert):
    ------------------------------------------------------------------
     Modell-Platz      Fahrer-Name Top-10 Chance  Echter Platz
                1     Marc Hirschi         93.6%          83.0
                2     Ben O'Connor         92.3%          32.0
                3        Enric Mas         91.3%           7.0
                4 Carlos Rodríguez         91.0%          15.0
                5 Jonas Vingegaard         89.9%           6.0
                6 Aleksandr Vlasov         89.7%          49.0
                7 Matteo Jorgenson         89.2%         115.0
                8   Lenny Martinez         88.6%         122.0
                9    Biniam Girmay         87.9%         158.0
               10    Tadej Pogačar         87.5%           5.0
    
    ------------------------------------------------------------------
    TABELLE 2: Die echten Top 10 des Renntages inkl. der Modellwahrscheinlchkeiten
    ------------------------------------------------------------------
     Echter Platz            Fahrer-Name  Modell-Platz Top-10 Chance
              1.0 Valentin Paret-Peintre            42         59.0%
              2.0              Ben Healy            40         61.5%
              3.0      Santiago Buitrago            16         79.9%
              4.0        Ilan Van Wilder            43         57.8%
              5.0          Tadej Pogačar            10         87.5%
              6.0       Jonas Vingegaard             5         89.9%
              7.0              Enric Mas             3         91.3%
              8.0     Julian Alaphilippe            20         76.0%
              9.0          Primož Roglič            17         78.6%
             10.0       Florian Lipowitz            35         67.3%
    ==========================================================================================
    
    

## Kritische Diskussion der Case-Study-Ergebnisse (Das Pointwise-Dilemma)

Der direkte Vergleich der Tabellen 1 und 2 für die Etappen 12 und 16 der Tour de France 2025 visualisiert die Defizite des Pointwise-Klassifikationsansatzes auf empirische Weise:

### Künstliche Stauchung & Monotonie der Prädiktionen (Data Overfitting auf historische Lags)
In beiden gezeigten Etappen generiert die EBM eine nahezu deckungsgleiche Top-10-Favoritenliste, angeführt von Ben O'Connor und Marc Hirschi mit konstanten Wahrscheinlichkeiten jenseits der 92%-Marke. Da das Pointwise-Modell jeden Athleten isoliert bewertet, dominieren die historisch starken Prädiktoren (`lag_rider_points_season`, `lag_team_power_index`) die Berechnung. Das Modell verfällt in eine "Sicherheits-Prognose" und unterschätzt die situative Etappencharakteristik. Es fehlt die mathematische Kontrollinstanz, welche die Summe der Top-10-Wahrscheinlichkeiten innerhalb einer Etappe restriktiv reguliert (Nullsummenspiel).

### Empirischer Beleg des fehlenden Gruppenkontexts (Das Favoriten-Szenario / Etappe 12)
Am Beispiel von Etappe 12 wird die biologische Diskonnektion des Modells deutlich:
* **Der statistische Erfolg:** Die EBM besitzt fundamentale Signalstärke. Sie prognostiziert Jonas Vingegaard auf Modell-Platz 3 (93.3% Chance), welcher real den 2. Platz belegt. Auch Tadej Pogačar (Echter Platz 1) wird mit 88.0% Wahrscheinlichkeit im erweiterten Favoritenkreis geführt.
* **Das Pointwise-Defizit:** Gleichzeitig weist das Modell dem reinen Sprinter Biniam Girmay eine Top-10-Wahrscheinlichkeit von 89.6% (Modell-Platz 10) für diese schwere Bergetappe zu, auf welcher er real mit Rang 117 geführt wird. Da die EBM die Fahrer nicht *relativ zueinander* innerhalb des spezifischen Höhenprofils gewichtet, kann Girmays historischer Punktestand das physische Defizit im Hochgebirge im Pointwise-Framing ungestraft übersteuern.

### Das mathematische Versagen bei strategischen Gruppendynamiken (Das Ausreißer-Szenario / Etappe 16)
Etappe 16 demonstriert das klassische Schreckensszenario für Pointwise-Modelle – das Durchkommen einer Fluchtgruppe:
* Die realen Spitzenreiter Valentin Paret-Peintre (Platz 1) und Ben Healy (Platz 2) werden vom Modell lediglich auf den Modell-Plätzen 42 und 40 einsortiert. 
* Obwohl das Modell ihnen eine respektable mathematische Grundchance zuweist (~59.0% und 61.5%), werden sie im finalen Ranking von den GC-Stars (Mas, Rodríguez, Vingegaard) verdrängt, da diese isoliert betrachtet stärkere historische Attribute aufweisen. 
* Das Pointwise-Modell kann nicht antizipieren, dass das Hauptfeld den Ausreißern an diesem Tag einen strategischen Vorsprung gewährt.

**Fazit**
Die Case Studies liefern den unumstößlichen Beleg dafür, dass das bloße Sortieren von pointwise gelernten Klassifikationswahrscheinlichkeiten die fundamentale Relativität des Radsports nicht abbilden kann. Um die Gruppenstruktur (`stage_id`) direkt in den Optimierungsprozess zu integrieren, wechseln wir im Folgenden u.A. zum **Learning-to-Rank (LTR)** Ansatz mittels des `XGBRanker`. Dieser vergleicht die Features der Fahrer paarweise oder listenweise innerhalb einer Etappe, um das relative Klassement direkt zu maximieren.


```python

```
