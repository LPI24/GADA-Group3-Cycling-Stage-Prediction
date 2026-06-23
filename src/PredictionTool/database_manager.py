#database_manager.py
# Der database_manager fungiert als zentraler Türsteher und Übersetzer, der die Verbindung zu deiner lokalen SQLite-Datenbank eigenständig öffnet, absichert und wieder schließt. Er nimmt einfache Python-Befehle entgegen und wandelt sie im Hintergrund in optimierte SQL-Abfragen um, sodass du dich im Hauptcode nicht mehr um Datenbank-Verbindungen kümmern musst. Am Ende liefert er dir die angeforderten Fahrer- oder Saisondaten blitzschnell als fertiges Pandas-DataFrame zurück, mit dem deine Live-Pipeline sofort weiterrechnen kann.


import sqlite3
import os
import pandas as pd

class CyclingDatabase:
    def __init__(self, db_path="../../data/databases/cycling_production.db"):
        """Initialisiert den Zugriffspfad auf die lokale SQLite-Datenbank."""
        self.db_path = db_path

    def get_connection(self):
        """Erstellt eine frische Verbindung zur SQLite-Datenbank."""
        return sqlite3.connect(self.db_path)

    def get_rider_master(self, rider_url):
        """Holt die zeitstabilen Stammdaten (Biometrie, Geburt) eines Fahrers."""
        conn = self.get_connection()
        query = "SELECT * FROM r_master WHERE rider_url = ?"
        df = pd.read_sql_query(query, conn, params=(rider_url,))
        conn.close()
        return df if not df.empty else None

    def get_rider_lags(self, rider_url, season_year):
        """Holt die historischen Saison-Lags (Punkte, Rang, Team-Tier) eines Fahrers."""
        conn = self.get_connection()
        query = "SELECT * FROM lags_historical WHERE rider_url = ? AND season_year = ?"
        df = pd.read_sql_query(query, conn, params=(rider_url, season_year))
        conn.close()
        return df if not df.empty else None

    def query(self, sql_string, params=None):
        """Führt eine völlig freie SQL-Abfrage aus und gibt ein DataFrame zurück."""
        conn = self.get_connection()
        df = pd.read_sql_query(sql_string, conn, params=params)
        conn.close()
        return df
