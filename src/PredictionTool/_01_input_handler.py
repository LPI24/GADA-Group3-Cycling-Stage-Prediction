# input_handler.py
import re

def get_user_inputs():
    print("==================================================================")
    print(" Abfrage welches Etappenergebnis soll voraus gesagt werden")
    print("==================================================================")

    # RENNEN ABFRAGEN & MAPPPEN
    races_mapping = {
        "giro": "giro-d-italia",
        "tour": "tour-de-france",
        "vuelta": "vuelta-a-espana"
    }

    while True:
        race_input = input("Welches Rennen? (Giro, Tour, Vuelta): ").strip().lower()
        if race_input in races_mapping:
            race_slug = races_mapping[race_input]
            break
        print("!!!Ungültige Eingabe! Bitte tippe entweder 'Giro', 'Tour' oder 'Vuelta'.")

    # JAHR ABFRAGEN
    while True:
        year_input = input("Welches Jahr?: ").strip()
        if year_input.isdigit():
            race_year = int(year_input)
            break
        print("!!!Ungültige Eingabe! Bitte gib eine Jahreszahl ein, die 2026 oder höher ist.")

    # ETAPPE ABFRAGEN (Zahlen von 1 bis 21)
    while True:
        stage_input = input("Welche Etappe? (Zahl von 1-21): ").strip()
        if stage_input.isdigit() and 1 <= int(stage_input) <= 21:
            stage_num = int(stage_input)
            break
        print("!!!Ungültige Eingabe! Grand Tours haben nur die Etappen 1 bis 21.")

    # LINK DYNAMISCH ZUSAMMENBAUEN
    # Muster: https://www.procyclingstats.com/race/tour-de-france/2026/stage-1
    pcs_stage_url = f"https://www.procyclingstats.com/race/{race_slug}/{race_year}/stage-{stage_num}"

    print(f"Gewähltes Rennen: {race_slug.replace('-', ' ').title()}")
    print(f"Saison-Jahr:     {race_year}")
    print(f"Etappennummer:   {stage_num}")
    print(f"Generierte URL:  {pcs_stage_url}")


    # Rückgabe als Dictionary, das direkt an den Scraper übergeben werden kann
    return {
        "race_slug": race_slug,
        "race_year": race_year,
        "stage_num": stage_num,
        "pcs_url": pcs_stage_url
    }

if __name__ == "__main__":
    get_user_inputs()
