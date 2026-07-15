# Cycling Stage Prediction

## Projektüberblick

Dieses Projekt untersucht die Vorhersage von Etappenplatzierungen im professionellen Straßenradsport mithilfe verschiedener Machine-Learning-Modelle. Ziel ist es nicht nur, einzelne Fahrer als mögliche Top-Platzierte zu klassifizieren, sondern innerhalb jeder Etappe eine modellbasierte Rangliste potenzieller Spitzenfahrer zu erzeugen.

Dazu werden historische Fahrer-, Team-, Renn- und Etappendaten aufbereitet und zu einer finalen Feature-Matrix zusammengeführt. Die Modelle werden chronologisch trainiert und evaluiert: Frühere Saisons dienen als Trainingsdaten, die Saison 2023 als Validierungszeitraum und die Saisons 2024/2025 als finales Testset.

Im Fokus steht der Vergleich mehrerer Modellansätze:

- Explainable Boosting Machine / GA²M
- TabPFN
- XGBoost Classifier
- XGBRanker

Die Modelle werden anhand rankingbasierter Metriken wie `NDCG@k`, `MAP@k`, `Winner Hit@k` und Spearman-Rangkorrelation bewertet. Dadurch wird geprüft, wie gut die Modelle relevante Fahrer innerhalb einer Etappe nach oben sortieren und ob sie den tatsächlichen Etappensieger zumindest innerhalb eines erweiterten Favoritenkreises platzieren.

Zusätzlich enthält das Repository ein prototypische Prediction-CLI-Tool. Diese ermöglicht es, für eine ausgewählte Grand-Tour-Etappe automatisch Etappendaten und Startlisten zu laden, fehlende Fahrerinformationen zu ergänzen, eine Feature-Matrix zu erzeugen und die gespeicherten Modelle auf die Etappe anzuwenden.
## Repository-Struktur

Das Repository ist in Datenordner, Modellartefakte, Notebooks und den prototypischen Prediction-Workflow gegliedert. Die folgende Übersicht zeigt die zentrale Ordnerstruktur des Projekts:

```text
GADA-Group3-Cycling-Stage-Prediction/
│
├── .vscode/                  # VS-Code-spezifische Einstellungen
│
├── data/
│   ├── charts/               # erzeugte Diagramme und Visualisierungen
│   ├── databases/            # lokale SQLite-Datenbank für Prediction Tool
│   ├── interim/              # Zwischenergebnisse während der Datenaufbereitung
│   ├── model_data/           # modellfertige Datenmatrizen und Splits
│   ├── models/               # gespeicherte Modellartefakte und Ergebnisdateien
│   ├── presentation/         # Zwischenpräsentationen und finale Präsentation
│   ├── processed/            # final bereinigte und zusammengeführte Datensätze
│   └── raw/                  # Rohdaten und ursprüngliche Eingabedaten (zu groß für GitHub)
│
├── src/
│   ├── Notebooks/            # Jupyter-Notebooks für Datenaufbereitung, Modelltraining und Evaluation
│   ├── PredictionTool/       # prototypische Anwendung zur Modellnutzung (CLI Tool)
│   └── scraping/code/        # Scraper Code
│
├── .gitignore                # Ausschlussregeln für Git
└── DOCUMENTATION.md          # Übersucht Feature- und Metadaten
```

## Installation und Environment

Das Projekt wurde mit Python `3.12` im Conda-Environment `gada` entwickelt. Die vollständige Paketumgebung ist in der Datei `environment.yml` dokumentiert. Dadurch kann das verwendete Environment reproduzierbar neu erstellt werden.

### Environment erstellen

```bash
conda env create -f environment.yml
conda activate gada

```


## Datenbasis


Die Datenbasis des Projekts besteht aus historischen Renn-, Etappen-, Fahrer- und Teamdaten aus dem professionellen Straßenradsport (Quelle: https://www.procyclingstats.com/). Dabei wird sich auf die drei großen Landesrundfahrten (Giro d'Italia, Tour de France und Vuelta a Espana) beschränkt. Die Daten werden in mehreren Schritten bereinigt, zusammengeführt und zu einer finalen Modellmatrix verarbeitet. Dabei werden unter anderem Etappenmerkmale, Fahrerinformationen, Teamindikatoren, Wetterdaten sowie Ranglistendaten berücksichtigt.

Die finale Modellierung basiert auf einem chronologischen Split:

```text
Trainingsdaten:      Saison 2005 bis einschließlich 2022
Validierungsdaten:   Saison 2023
Testdaten:           Saisons 2024 und 2025
```

## Notebook-Reihenfolge

Die Modellierung und Evaluation erfolgt primär über Jupyter-Notebooks im Ordner `src/Notebooks/`. Die Notebooks sind größtenteils nummeriert und bilden den Projektablauf von der Datenaufbereitung über Feature Engineering bis zur finalen Modellbewertung ab.

Für die vollständige Reproduktion des Projekts sollte die folgende Reihenfolge eingehalten werden:

| Schritt | Notebook | Zweck |
|---:|---|---|
| 1 | `01_Data_Merging_and_Cleaning.ipynb` | Zusammenführung und erste Bereinigung der Ausgangsdaten |
| 2 | `03_01_Koordinaten.ipynb` / `03_02_Koordinaten.ipynb` | Aufbereitung und Ergänzung geografischer Koordinaten |
| 3 | `04_weather.ipynb` | Einbindung und Aufbereitung von Wetterdaten |
| 4 | `05_Data_Cleaning_Exploration.ipynb` | explorative Prüfung und weitere Datenbereinigung |
| 5 | `06_Missing_Values_Transformation.ipynb` bis `06_13_von_how_analysis.ipynb` | Behandlung fehlender Werte, Transformationen und Feature-Vorbereitung |
| 6 | `07_01_extract_clean_gradient_final_km.ipynb` | Extraktion und Bereinigung des finalen Steigungsmerkmals |
| 7 | `07_02_Feature_Selection_Dropping.ipynb` | Auswahl relevanter Features und Entfernen nicht benötigter Spalten |
| 8 | `08_00_transformation_dtypes.ipynb` | Vereinheitlichung der Datentypen |
| 9 | `08-00-01_Data_Leakage_Ranking_Team_Driver.ipynb` | Erstellung Lag-Ranglisten-Features zur Vermeidung von Data Leakage |
| 10 | `08_01_Univariate_Analysis.ipynb` | univariate Analyse der finalen Features |
| 11 | `08_02_Bivariate_Analysis.ipynb` | bivariate Analyse der finalen Features |
| 12 | `10-00_Model_Data_Prep.ipynb` | Erstellung der finalen Modelldatenbasis mit Trainings-, Validierungs- und Testsplit |
| 13 | `10-01_EBM_classif.ipynb` | Training, Tuning und Evaluation des EBM-/GA²M-Modells |
| 14 | `11-01_XGBoost_Classifier.ipynb` / `11-02_XGBoost_Ranker.ipynb` | Training und Evaluation des XGBoost-Klassifikators sowie XGBoost Ranker |
| 15 | `12-00_TabPFN_Intro_Setup.ipynb` bis `12-04_TabPFN_SHAP_Interpretation.ipynb` | Setup, Tuning, finale Evaluation und Interpretation des TabPFN-Modells |

### Unterstützende und explorative Notebooks

Einige Notebooks dienen vor allem der Exploration, methodischen Einordnung oder zum Testen einzelner Verarbeitungsschritte. Sie sind hilfreich für das Verständnis des Projekts, aber nicht zwingend erforderlich, wenn die vorbereiteten Daten bereits vorhanden sind.

Dazu gehören insbesondere:

```text
02_Model_Explanations.ipynb
05_01_Data_Exploration.ipynb
10_Feature_Selection.md

```

## Nutzung des CLI Prediction Tools

Neben der notebookbasierten Modellierung enthält das Projekt ein prototypisches CLI Prediction Tool. Das Tool ermöglicht es, für eine ausgewählte Grand-Tour-Etappe automatisch eine Startliste und Etappeninformationen zu laden, fehlende Fahrer- und Lag-Ranglisten-Daten zu ergänzen, eine Feature-Matrix zu erzeugen und anschließend die gespeicherten Modelle auf die Etappe anzuwenden.


Der Einstiegspunkt des Tools ist:

```text
src/scraping/code/_00_Pipeline.py
```

### Starten des Tools
Das Tool wird aus dem Projektordner heraus gestartet:

```bash
cd src/scraping/code
python _00_Pipeline.py
```

### Ablauf

Nach dem Start fragt das Tool interaktiv die gewünschte Etappe ab. Unterstützt werden die drei Grand Tours:

- Giro
- Tour
- Vuelta

Der typische Ablauf ist:

1. Auswahl von Rennen, Jahr und Etappe
2. Automatische Erstellung der ProCyclingStats-URL
3. Scraping der Etappenmetadaten und Startliste
4. Prüfung der Fahrer gegen die lokale SQLite-Datenbank
5. Nachscraping fehlender Fahrerprofile und Lag-Ranglisten-Daten
6. Erstellung der Feature-Matrix
7. Anwendung der gespeicherten Modelle
8. Ausgabe der prognostizierten Ranglisten und, falls reale Ergebnisse vorhanden sind, der Evaluationsmetriken

Im Pipeline-Code werden die Startliste und Etappenmetadaten zunächst über den Scraper geladen. Falls keine gültigen Daten zurückgegeben werden, wird der Durchlauf abgebrochen. Anschließend werden fehlende Fahrerprofile und Lag-Ranglisten-Daten geprüft und bei Bedarf ergänzt.

## Benötigte Dateien

Für die Nutzung des CLI Prediction Tools müssen folgende Artefakte vorhanden sein:

- `data/databases/cycling_production.db`
- `data/models/ebm_best_binary_ensemble.pkl`
- `data/models/xgboost_classifier_results.json`
- `data/models/tabpfn_final_results.csv`

Die SQLite-Datenbank wird vom Tool genutzt, um Fahrerprofile und Lag-Ranglisten-Features mit der aktuellen Startliste zu verbinden. Im Code wird dazu die Datenbank unter `data/databases/cycling_production.db` geladen und mit den Fahrer-URLs der Startliste abgeglichen.

### Ausgabe

Das Tool gibt zunächst eine konsolidierte Startliste mit Fahrerinformationen und Vorjahres-Lag-Ranglisten-Daten aus. Danach wird die Feature-Matrix erzeugt und an die Modellinferenz übergeben. Falls reale Etappenergebnisse verfügbar sind, werden diese mit den Fahrer-URLs gemappt und für eine Backtest-Auswertung verwendet.

Die Ausgabe umfasst je nach verfügbarer Datenlage:

- prognostizierte Top-Fahrer pro Modell
- Modell-Scores
- reale Platzierungen, falls die Etappe bereits abgeschlossen ist
- NDCG@5, NDCG@10 und NDCG@20
- AP@10
- Winner Hit@1, Winner Hit@5, Winner Hit@10 und Winner Hit@20

### Hinweis

Das CLI Prediction Tool ist als prototypische Anwendung zu verstehen. Die wissenschaftliche Modellbewertung erfolgt primär über die Notebooks in `src/Notebooks/`. Das Tool dient ergänzend dazu, den entwickelten Workflow auf einzelne Etappen anzuwenden und die Modellvorhersagen praktisch nutzbar zu machen.


## Outputs


Während der Datenaufbereitung, Modellierung und Evaluation erzeugt das Projekt verschiedene Output-Dateien. Diese werden je nach Schritt in den Ordnern `data/model_data/`, `data/models/`, `data/charts/` und `data/processed/` abgelegt.

### Datenoutputs

Die finalen Modelldaten werden im Ordner `data/model_data/` gespeichert. Sie enthalten die vorbereiteten Feature-Matrizen, Zielvariablen, Metadaten und chronologischen Splits für Training, Validierung und Test.

Typische Inhalte sind:

- `data/model_data/` – finale Modellmatrizen, Zielvariablen und Splits
- `data/processed/` – bereinigte und zusammengeführte Datensätze
- `data/interim/` – Zwischenergebnisse einzelner Aufbereitungsschritte

Die Modelldaten bilden die Grundlage für die finalen Modellnotebooks. Wenn diese Dateien bereits vorhanden sind, müssen die frühen Datenaufbereitungsnotebooks nicht erneut vollständig ausgeführt werden.

### Modellartefakte

Trainierte Modelle und gespeicherte Ergebnisdateien werden im Ordner `data/models/` abgelegt. Diese Dateien werden entweder durch die Modellnotebooks erzeugt oder vom CLI Prediction Tool geladen.
Wichtige Modellartefakte sind:

- `data/models/ebm_best_binary_ensemble.pkl` – gespeichertes EBM-/GA²M-Ensemble
- `data/models/xgboost_classifier_results.json` – gespeicherte XGBoost-Ergebnisse
- `data/models/tabpfn_final_results.csv` – gespeicherte TabPFN-Ergebnisse


### Visualisierungen und Auswertungstabellen

Diagramme und tabellarische Auswertungen werden vor allem im Ordner `data/charts/` gespeichert. Dazu gehören unter anderem Zielverteilungen, ROC-Kurven und Feature-Importance-Auswertungen.

Die erzeugten Visualisierungen dienen sowohl der Modellinterpretation als auch der Dokumentation der Ergebnisse im Projektbericht.

### Hinweis

Nicht alle Outputs werden automatisch versioniert. Große Rohdaten, temporäre Zwischenergebnisse oder lokal erzeugte Dateien können durch `.gitignore` vom Repository ausgeschlossen sein. Für die Reproduktion sind insbesondere die vorbereiteten Modelldaten in `data/model_data/` und die gespeicherten Modellartefakte in `data/models/` relevant.

## Hinweise zur Reproduzierbarkeit

Für eine möglichst vollständige Reproduktion der Ergebnisse sollte das Projekt in einem frischen Clone des GitHub-Repositories ausgeführt werden. Dadurch wird sichergestellt, dass keine lokalen Änderungen, temporären Dateien oder alte Modellartefakte die Ergebnisse beeinflussen.

### Empfohlener Ablauf

```bash
git clone https://github.com/LPI24/GADA-Group3-Cycling-Stage-Prediction.git
cd GADA-Group3-Cycling-Stage-Prediction
```

Anschließend wird das Conda-Environment aus der bereitgestellten `environment.yml` erstellt:

```bash
conda env create -f environment.yml
conda activate gada
```

Falls das Environment bereits existiert, kann es aktualisiert werden:

```bash
conda env update -f environment.yml --prune
conda activate gada
```

### Reproduktionsreihenfolge

Für eine vollständige Reproduktion aus den vorbereiteten Modelldaten sollten zuerst die finalen Modelldaten und anschließend die Modellnotebooks ausgeführt werden:


Wenn die Rohdaten vollständig vorliegen und der gesamte Datenaufbereitungsprozess reproduziert werden soll, müssen zusätzlich die vorherigen Cleaning-, Feature-Engineering- und Data-Leakage-Notebooks in der dokumentierten Reihenfolge ausgeführt werden.
Dabei ist darauf zu achten, dass in einigen Notebooks händisch Daten bereinigt wurden und diese nur schwer reproduziert werden können. Entsprechende Hinweise sind in den jeweiligen Notebooks enthalten.

### Benötigte Daten und Artefakte

Einige Rohdaten sind aufgrund ihrer Größe nicht vollständig im Repository enthalten. Für die Reproduktion der finalen Modelle sind insbesondere die vorbereiteten Daten in `data/model_data/` sowie die gespeicherten Modellartefakte in `data/models/` relevant.

Das CLI Prediction Tool setzt zusätzlich voraus, dass die lokale SQLite-Datenbank und die benötigten Modellartefakte vorhanden sind:

- `data/databases/cycling_production.db`
- `data/models/ebm_best_binary_ensemble.pkl`
- `data/models/xgboost_classifier_results.json`
- `data/models/tabpfn_final_results.csv`

### Hinweise zu externen Abhängigkeiten

Teile der Datengewinnung basieren auf Scraping von ProCyclingStats. Die Verfügbarkeit und Struktur externer Webseiten kann sich ändern. Dadurch kann es vorkommen, dass Scraping-Notebooks oder das CLI Prediction Tool angepasst werden müssen, obwohl die bereits gespeicherten Modelldaten weiterhin reproduzierbar nutzbar sind.

TabPFN wurde im Projekt teilweise über gespeicherte Ergebnisdateien beziehungsweise externe Modellinferenz eingebunden. API-Keys oder andere Zugangsdaten werden nicht im Repository gespeichert und müssen bei Bedarf lokal konfiguriert werden.
