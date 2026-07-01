# TabPFN Case Study TDF 2025 Stage 12 und 16

## standard - tour-de-france_2025_ST12

Modell für diesen Vergleich: `TabPFNClassifier | variant=standard | score=score_topk_raw_sum`.

Parameter aus `12-02`: `{"average_before_softmax": true, "balance_probabilities": true, "n_estimators": 4, "softmax_temperature": 0.8}`.

Score: `score_topk_raw_sum = p_top5_raw + p_top10_raw + p_top20_raw`.

- Top20-Treffer: 8 von 20 predicted Top20-Fahrern; Precision 0.400, Recall 0.400.
- Echter Sieger: Tadej Pogačar mit vorhergesagtem Rang 1.
- Treffer in der vorhergesagten Top20: Tadej Pogačar, Jonas Vingegaard, Matteo Jorgenson, Ben O'Connor, Remco Evenepoel, Primož Roglič, Tobias Halland Johannessen, Oscar Onley.
- Überschätzte predicted Top20-Fahrer: Aleksandr Vlasov, Marc Hirschi, Adam Yates, Carlos Rodríguez, Enric Mas, Mattias Skjelmose, Alex Aranburu, Magnus Cort, Lenny Martinez, Alex Baudin, Julian Alaphilippe, Santiago Buitrago.
- Verpasste echte Top20-Fahrer: Florian Lipowitz, Kévin Vauquelin, Felix Gall, Cristián Rodríguez, Einer Rubio, Raúl García Pierna, Guillaume Martin, Sergio Higuita, Bruno Armirail, Jhonatan Narváez, Jordan Jegat, Aurélien Paret-Peintre.

Die vollständige Position-zu-Position-Tabelle steht in `tabpfn_case_study_tdf2025_stage12_16_top20_comparison.csv`.

## standard - tour-de-france_2025_ST16

Modell für diesen Vergleich: `TabPFNClassifier | variant=standard | score=score_topk_raw_sum`.

Parameter aus `12-02`: `{"average_before_softmax": true, "balance_probabilities": true, "n_estimators": 4, "softmax_temperature": 0.8}`.

Score: `score_topk_raw_sum = p_top5_raw + p_top10_raw + p_top20_raw`.

- Top20-Treffer: 10 von 20 predicted Top20-Fahrern; Precision 0.500, Recall 0.500.
- Echter Sieger: Valentin Paret-Peintre mit vorhergesagtem Rang 23.
- Treffer in der vorhergesagten Top20: Tadej Pogačar, Jonas Vingegaard, Adam Yates, Enric Mas, Carlos Rodríguez, Alex Aranburu, Primož Roglič, Santiago Buitrago, Julian Alaphilippe, Oscar Onley.
- Überschätzte predicted Top20-Fahrer: Aleksandr Vlasov, Matteo Jorgenson, Marc Hirschi, Ben O'Connor, Magnus Cort, Alex Baudin, Lenny Martinez, Tobias Halland Johannessen, Jenno Berckmoes, Biniam Girmay.
- Verpasste echte Top20-Fahrer: Valentin Paret-Peintre, Ben Healy, Ilan Van Wilder, Florian Lipowitz, Gregor Mühlberger, Thymen Arensman, Xandro Meurisse, Felix Gall, Raúl García Pierna, Kévin Vauquelin.

Die vollständige Position-zu-Position-Tabelle steht in `tabpfn_case_study_tdf2025_stage12_16_top20_comparison.csv`.

## thinking_medium_roc_auc - tour-de-france_2025_ST12

Modell für diesen Vergleich: `TabPFNClassifier | variant=thinking_medium_roc_auc | score=score_topk_raw_sum`.

Parameter aus `12-02`: `{"average_before_softmax": true, "balance_probabilities": true, "n_estimators": 4, "softmax_temperature": 0.8}`.

Score: `score_topk_raw_sum = p_top5_raw + p_top10_raw + p_top20_raw`.

- Top20-Treffer: 6 von 20 predicted Top20-Fahrern; Precision 0.300, Recall 0.300.
- Echter Sieger: Tadej Pogačar mit vorhergesagtem Rang 1.
- Treffer in der vorhergesagten Top20: Tadej Pogačar, Jonas Vingegaard, Matteo Jorgenson, Ben O'Connor, Remco Evenepoel, Primož Roglič.
- Überschätzte predicted Top20-Fahrer: Marc Hirschi, Aleksandr Vlasov, Enric Mas, Adam Yates, Carlos Rodríguez, Biniam Girmay, Lenny Martinez, Mattias Skjelmose, Santiago Buitrago, Julian Alaphilippe, Magnus Cort, Tim Merlier, Jenno Berckmoes, Romain Grégoire.
- Verpasste echte Top20-Fahrer: Florian Lipowitz, Tobias Halland Johannessen, Oscar Onley, Kévin Vauquelin, Felix Gall, Cristián Rodríguez, Einer Rubio, Raúl García Pierna, Guillaume Martin, Sergio Higuita, Bruno Armirail, Jhonatan Narváez, Jordan Jegat, Aurélien Paret-Peintre.

Die vollständige Position-zu-Position-Tabelle steht in `tabpfn_case_study_tdf2025_stage12_16_top20_comparison.csv`.

## thinking_medium_roc_auc - tour-de-france_2025_ST16

Modell für diesen Vergleich: `TabPFNClassifier | variant=thinking_medium_roc_auc | score=score_topk_raw_sum`.

Parameter aus `12-02`: `{"average_before_softmax": true, "balance_probabilities": true, "n_estimators": 4, "softmax_temperature": 0.8}`.

Score: `score_topk_raw_sum = p_top5_raw + p_top10_raw + p_top20_raw`.

- Top20-Treffer: 10 von 20 predicted Top20-Fahrern; Precision 0.500, Recall 0.500.
- Echter Sieger: Valentin Paret-Peintre mit vorhergesagtem Rang 25.
- Treffer in der vorhergesagten Top20: Tadej Pogačar, Jonas Vingegaard, Enric Mas, Adam Yates, Carlos Rodríguez, Primož Roglič, Julian Alaphilippe, Santiago Buitrago, Alex Aranburu, Oscar Onley.
- Überschätzte predicted Top20-Fahrer: Matteo Jorgenson, Marc Hirschi, Aleksandr Vlasov, Ben O'Connor, Biniam Girmay, Lenny Martinez, Magnus Cort, Tim Merlier, Alex Baudin, Romain Grégoire.
- Verpasste echte Top20-Fahrer: Valentin Paret-Peintre, Ben Healy, Ilan Van Wilder, Florian Lipowitz, Gregor Mühlberger, Thymen Arensman, Xandro Meurisse, Felix Gall, Raúl García Pierna, Kévin Vauquelin.

Die vollständige Position-zu-Position-Tabelle steht in `tabpfn_case_study_tdf2025_stage12_16_top20_comparison.csv`.

