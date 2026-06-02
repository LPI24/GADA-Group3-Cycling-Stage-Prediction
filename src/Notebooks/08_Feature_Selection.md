| Spaltenname | Kategorie / Verwendung | Anmerkung / Begründung für das Modell |
| :--- | :--- | :--- |
| **race** | MetaData | Nicht für das Training; ID zur Identifikation der Datenzeile. |
| **year** | MetaData | Nicht für das Training; ID zur zeitlichen Zuordnung. |
| **url** | MetaData | Nicht für das Training; Eindeutiger Identifikator der Etappe. |
| **rank** | **Target Feature (`y`)** | Das offizielle Endergebnis des Fahrers (Zielvariable). |
| **rider_url** | MetaData | Nicht für das Training; ID des Fahrers zur Rückverfolgung. |
| **time_gap** | Löschen (Data Leakage) | Enthält implizit das Rennergebnis; mathematisch unzulässig vor dem Rennen. |
| **birthdate** | Löschen | Redundant, da durch das Feature `age_at_race` bereits präzise erfasst. |
| **height** | Fahrer Physis | Biometrisches Feature (Einfluss auf Aerodynamik und Fahrertyp). |
| **name** | MetaData | Klartextname des Fahrers; für den Menschen zur Analyse nach dem Training. |
| **nationality** | MetaData | Kategoriale ID; in diesem Fall als MetaData ausgeschlossen. |
| **weight** | Fahrer Physis | Biometrisches Feature (Entscheidend für Steigfähigkeit auf Bergen). |
| **url_name** | MetaData | Identifikator-String; nicht numerisch verwertbar. |
| **departure** | MetaData | Startort der Etappe |
| **arrival** | MetaData | Zielort der Etappe |
| **distance** | Rennprofil | Streckenlänge in km (Einfluss auf die Ausdauerkomponente). |
| **vertical_meters** | Rennprofil | Kumulierte Höhenmeter (Entscheidender Faktor für Kletterspezialisten). |
| **won_how** | Löschen (Data Leakage) | Gibt die Art des Ankunftsszenarios vorab preis (z. B. "Sprint"). |
| **avg_speed** | Löschen (Data Leakage) | Die Durchschnittsgeschwindigkeit steht erst nach dem Ziel fest. |
| **race_ranking** | Löschen | haben wir über den race_competitiveness_median erstellt |
| **one_day_races** | Spezialisierung | PCS-Punkteprofil für Eintagesrennen (Fahrertyp-Indikator). |
| **gc** | Spezialisierung | PCS-Punkteprofil für das Gesamtklassement (Klassementfahrer-Indikator). |
| **time_trial** | Spezialisierung | PCS-Punkteprofil für Zeitfahren (Zeitfahrspezialisten-Indikator). |
| **sprint** | Spezialisierung | PCS-Punkteprofil für Massensprints (Sprintspezialisten-Indikator). |
| **climber** | Spezialisierung | PCS-Punkteprofil für Hochgebirge (Kletterspezialisten-Indikator). |
| **hills** | Spezialisierung | PCS-Punkteprofil für hügeliges Terrain / Ardennen-Klassiker. |
| **stage_nr** | Rennprofil | Etappennummer (Faktor für kumulierte Ermüdung bei Rundfahrten). |
| **date** | MetaData | Renndatum |
| **departure_lat** | MetaData | Geokoordinate des Startorts; im aggregierten Wetter-Feature bereits impliziert. |
| **departure_lon** | MetaData | Geokoordinate des Startorts. |
| **arrival_lat** | MetaData | Geokoordinate des Zielorts. |
| **arrival_lon** | MetaData | Geokoordinate des Zielorts. |
| **rider_points_season** | Fahrer Stärke | Metrische Skala; quantifiziert den absoluten Leistungsabstand in der Weltspitze. |
| **rider_rank_season** | Fahrer Stärke | Ordinale Skala; liefert die exakte relative Hierarchie im Fahrerfeld. |
| **current_team** | MetaData | Klartextname des aktuellen Teams. |
| **departure_temp_mittel** | Drop / Entfernen | Ersetzt durch das entkoppelte Feature `weather_temp_mean`. |
| **departure_regen_mittel** | Drop / Entfernen | Ersetzt durch das entkoppelte Feature `weather_rain_prob_mean`. |
| **departure_wind_mittel** | Drop / Entfernen | Ersetzt durch das zyklisch berechnete Feature `wind_stability_index`. |
| **departure_luftfeuchte_mittel** | Drop / Entfernen | Ersetzt durch das entkoppelte Feature `weather_humidity_mean`. |
| **departure_niederschlag_mittel**| Drop / Entfernen | Ersetzt durch das entkoppelte Feature `weather_precipitation_mean`. |
| **departure_windrichtung_mittel**| Drop / Entfernen | Ersetzt durch das zyklisch berechnete Feature `wind_stability_index`. |
| **arrival_temp_mittel** | Drop / Entfernen | Ersetzt durch das Feature `weather_temp_trend`. |
| **arrival_regen_mittel** | Drop / Entfernen | Ersetzt durch das entkoppelte Feature `weather_rain_prob_mean`. |
| **arrival_wind_mittel** | Drop / Entfernen | Ersetzt durch das zyklisch berechnete Feature `wind_stability_index`. |
| **arrival_luftfeuchte_mittel** | Drop / Entfernen | Ersetzt durch das entkoppelte Feature `weather_humidity_mean`. |
| **arrival_niederschlag_mittel** | Drop / Entfernen | Ersetzt durch das entkoppelte Feature `weather_precipitation_mean`. |
| **arrival_windrichtung_mittel** | Drop / Entfernen | Ersetzt durch das zyklisch berechnete Feature `wind_stability_index`. |
| **time_gap_seconds** | Löschen (Data Leakage) | Perfekt korreliert mit dem Rang; führt zu fatalem Zielvariablen-Leak. |
| **team_tier** | Team Stärke | Kategoriale Einstufung der Team-Klasse (UCI WorldTeam, ProTeam etc.). |
| **age_at_race** | Fahrer Physis | Berechnetes Alter des Fahrers am Renntag (Erfahrung vs. Regeneration). |
| **rider_bmi** | Fahrer Physis | Kombiniertes biometrisches Feature ($Gewicht / Gr\ddot{o}\beta e^2$). |
| **race_competitiveness_median** | Rennprofil | Mediane Stärke des Fahrerfeldes (Misst die qualitative Tiefe des Rennens). |
| **team_power_index** | Team Stärke | Aggregierte Stärke der Teamkollegen auf dieser spezifischen Etappe. |
| **wind_speed_delta** | Drop / Zwischenschritt | Verarbeitet im finalen zusammengeführten Wind-Stabilitätsindex. |
| **wind_direction_delta** | Drop / Zwischenschritt | Verarbeitet im finalen zusammengeführten Wind-Stabilitätsindex. |
| **wind_stability_index** | **Wetter (ML Feature)** | Zyklisch berechneter Index (0 bis 1) für die atmosphärische Windruhe. |
| **weather_temp_mean** | **Wetter (ML Feature)** | Makroklimatisches Temperaturniveau (Mittelwert aus Start und Ziel). |
| **weather_temp_trend** | **Wetter (ML Feature)** | Meteorologischer linearer Trend (Ziel - Start) mit echtem Vorzeichen. |
| **weather_rain_prob_mean** | **Wetter (ML Feature)** | Aggregierte mittlere Regenwahrscheinlichkeit des Renntages. |
| **weather_precipitation_mean**| **Wetter (ML Feature)** | Aggregierte mittlere Niederschlagsmenge (Indikator für Schlammschlachten). |
| **weather_humidity_mean** | **Wetter (ML Feature)** | Mittlere relative Luftfeuchtigkeit (Einfluss auf die Thermoregulation). |
