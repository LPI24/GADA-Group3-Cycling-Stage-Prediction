# Data Dictionary - Cycling Prediction Project

Diese Tabelle beschreibt die flache Datenstruktur (Flat File) nach dem Preprocessing.

| Spalte | Datentyp | Beschreibung | Bedeutung für das Modell |
| :--- | :--- | :--- | :--- |
| **Identifikatoren** | | | |
| `race` | `object` | Name des Rennens (z.B. Tour de France). | Kategorisierung |
| `year` | `int64` | Das Jahr, in dem das Rennen stattfand. | Zeitliche Einordnung |
| `stage_nr` | `int64` | Nummer der Etappe innerhalb des Rennens. | Etappenkontext |
| `date` | `datetime` | Das exakte Datum des Renntages. | Zeitfeature / Alter-Berechnung |
| `url` | `object` | Link zur PCS-Etappenseite. | Metadaten (nicht für ML) |
| `name` | `object` | Vollständiger Name des Fahrers. | Identifikator |
| `rider_url` | `object` | Link zum PCS-Profil des Fahrers. | Eindeutige ID für Merges |
| `url_name` | `object` | URL-konformer Name des Fahrers. | Metadaten |
| **Ergebnisdaten (Labels)** | | | |
| `rank` | `float/int`| **Zielvariable:** Platzierung im Ziel. | Vorhersage-Label |
| `time_gap` | `object` | Zeitabstand zum Etappensieger. | Sekundäre Zielvariable |
| `won_how` | `object` | Art des Sieges (z.B. Sprint, Solo, Zeitfahren). | Renncharakter-Analyse |
| **Fahrer-Eigenschaften** | | | |
| `birthdate` | `datetime` | Geburtsdatum des Fahrers. | Berechnung `age_at_race` |
| `height` | `float64` | Körpergröße des Fahrers in cm. | Physisches Feature (BMI) |
| `weight` | `float64` | Gewicht des Fahrers in kg. | Physisches Feature (BMI) |
| `nationality` | `object` | Nationalität des Fahrers. | Kategorisches Feature |
| `current_team` | `object` | Team des Fahrers im jeweiligen Rennjahr. | Teamstärke / Support |
| `team_class` | `object` | Lizenzklasse des Teams (WT, PRT, PCT). | Qualitäts-Indikator |
| **Fahrer-Leistung (Saison)**| | | |
| `rider_points_season` | `float64` | UCI-Punkte des Fahrers in der aktuellen Saison. | **Stärkstes Feature** |
| `rider_rank_season` | `float64` | Weltranglistenplatz des Fahrers in der Saison. | Relative Stärke |
| **Fahrer-Spezialisierung**| (0-100 Skala) | | |
| `one_day_races` | `float64` | Score für Eintagesrennen. | Spezialisierung |
| `gc` | `float64` | Score für Gesamtwertung (General Classification). | Spezialisierung |
| `time_trial` | `float64` | Score für Zeitfahren. | Spezialisierung |
| `sprint` | `float64` | Score für Massensprints. | Spezialisierung |
| `climber` | `float64` | Score für reine Kletterer. | Spezialisierung |
| `hills` | `float64` | Score für hügeliges Terrain. | Spezialisierung |
| **Strecken-Profil** | | | |
| `distance` | `float/obj`| Gesamtlänge der Etappe in km. | Ausdauer-Faktor |
| `vertical_meters` | `float/obj`| Gesamte Höhenmeter der Etappe. | Schwierigkeitsgrad |
| `profile_score` | `float/obj`| PCS-Schwierigkeitsscore der Etappe. | Anforderungsprofil |
| `departure` / `arrival` | `object` | Start- und Zielort der Etappe. | Geografie |
| `departure_lat/lon` | `float64` | Koordinaten des Startorts. | Wetter-Merge / Geografie |
| `arrival_lat/lon` | `float64` | Koordinaten des Zielorts. | Wetter-Merge / Geografie |
| `race_ranking` | `object` | UCI-Kategorie des Rennens (z.B. 2.UWT). | Bedeutung des Rennens |
| **Wetterdaten (Start/Ziel)**| | | |
| `..._temp_mittel` | `float64` | Durchschnittstemperatur. | Wetter-Einfluss |
| `..._regen_mittel` | `float64` | Durchschnittliche Regenmenge. | Wetter-Einfluss |
| `..._wind_mittel` | `float64` | Durchschnittliche Windstärke. | Wetter-Einfluss |
| `..._luftfeuchte_mittel`| `float64` | Relative Luftfeuchtigkeit. | Wetter-Einfluss |
| `..._niederschlag_mittel`| `float64` | Niederschlagswahrscheinlichkeit. | Wetter-Einfluss |
| `..._windrichtung_mittel`| `float64` | Windrichtung in Grad. | Wetter-Einfluss |
| **Sonstiges** | | | |
