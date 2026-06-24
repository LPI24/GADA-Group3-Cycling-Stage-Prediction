import pandas as pd
from database_manager import CyclingDatabase


DRY_RUN = False
# Erst True lassen.
# Wenn die Vorschau passt: auf False setzen und nochmal ausführen.


RIDERS = [
    {
        "name": "Ghebreigzabhier Amanuel",
        "slug": "amanuel-ghebreigzabhier",
        "alt_names": ["Amanuel Ghebreigzabhier", "Ghebreigzabhier Amanuel"],
    },
    {
        "name": "Leknessund Andreas",
        "slug": "andreas-leknessund",
        "alt_names": ["Andreas Leknessund", "Leknessund Andreas"],
    },
    {
        "name": "Plowright Jensen",
        "slug": "jensen-plowright",
        "alt_names": ["Jensen Plowright", "Plowright Jensen"],
    },
    {
        "name": "Vergallito Luca",
        "slug": "luca-vergallito",
        "alt_names": ["Luca Vergallito", "Vergallito Luca"],
    },
    {
        "name": "Bayer Tobias",
        "slug": "tobias-bayer",
        "alt_names": ["Tobias Bayer", "Bayer Tobias"],
    },
    {
        "name": "Groenewegen Dylan",
        "slug": "dylan-groenewegen",
        "alt_names": ["Dylan Groenewegen", "Groenewegen Dylan"],
    },
    {
        "name": "Holter Ådne",
        "slug": "adne-holter",
        "alt_names": ["Ådne Holter", "Adne Holter", "Holter Ådne", "Holter Adne"],
    },
    {
        "name": "de Jong Timo",
        "slug": "timo-de-jong",
        "alt_names": ["Timo de Jong", "de Jong Timo", "De Jong Timo"],
    },
    {
        "name": "Groves Kaden",
        "slug": "kaden-groves",
        "alt_names": ["Kaden Groves", "Groves Kaden"],
    },
    {
        "name": "Price-Pejtersen Johan",
        "slug": "johan-price-pejtersen",
        "alt_names": ["Johan Price-Pejtersen", "Price-Pejtersen Johan"],
    },
    {
        "name": "Christen Fabio",
        "slug": "fabio-christen",
        "alt_names": ["Fabio Christen", "Christen Fabio"],
    },
    {
        "name": "Blikra Erlend",
        "slug": "erlend-blikra",
        "alt_names": ["Erlend Blikra", "Blikra Erlend"],
    },
    {
        "name": "Moschetti Matteo",
        "slug": "matteo-moschetti",
        "alt_names": ["Matteo Moschetti", "Moschetti Matteo"],
    },
]


def get_all_tables(conn):
    query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """
    return pd.read_sql_query(query, conn)["name"].tolist()


def get_table_columns(conn, table_name):
    df_cols = pd.read_sql_query(f"PRAGMA table_info({table_name})", conn)
    return df_cols["name"].tolist()


def resolve_rider_urls(conn):
    slugs = [r["slug"] for r in RIDERS]

    all_names = []
    for rider in RIDERS:
        all_names.append(rider["name"])
        all_names.extend(rider["alt_names"])

    all_names_lower = sorted(set(name.lower() for name in all_names))
    slugs_lower = sorted(set(slug.lower() for slug in slugs))

    slug_placeholders = ",".join(["?"] * len(slugs_lower))
    name_placeholders = ",".join(["?"] * len(all_names_lower))

    query = f"""
        SELECT
            rider_url,
            meta_name,
            meta_url_name,
            nationality,
            birthdate
        FROM r_master
        WHERE LOWER(rider_url) IN ({slug_placeholders})
           OR LOWER(meta_url_name) IN ({slug_placeholders})
           OR LOWER(meta_name) IN ({name_placeholders})
        ORDER BY rider_url
    """

    params = slugs_lower + slugs_lower + all_names_lower

    df_found = pd.read_sql_query(query, conn, params=params)

    found_urls = set(df_found["rider_url"].tolist()) if not df_found.empty else set()

    # Auch Slugs aufnehmen, die vielleicht nicht in r_master stehen,
    # aber in anderen Tabellen vorhanden sein könnten.
    target_urls = sorted(set(slugs) | found_urls)

    return target_urls, df_found


def preview_counts(conn, target_urls):
    tables = get_all_tables(conn)

    rows = []

    for table in tables:
        columns = get_table_columns(conn, table)

        if "rider_url" not in columns:
            continue

        placeholders = ",".join(["?"] * len(target_urls))

        query = f"""
            SELECT COUNT(*) AS count_rows
            FROM {table}
            WHERE rider_url IN ({placeholders})
        """

        count = pd.read_sql_query(query, conn, params=target_urls)["count_rows"].iloc[0]

        if count > 0:
            rows.append(
                {
                    "table_name": table,
                    "rows_to_delete": int(count),
                }
            )

    return pd.DataFrame(rows)


def delete_riders(conn, target_urls):
    cursor = conn.cursor()
    tables = get_all_tables(conn)

    delete_log = []

    placeholders = ",".join(["?"] * len(target_urls))

    for table in tables:
        columns = get_table_columns(conn, table)

        if "rider_url" not in columns:
            continue

        cursor.execute(
            f"""
            DELETE FROM {table}
            WHERE rider_url IN ({placeholders})
            """,
            target_urls
        )

        if cursor.rowcount > 0:
            delete_log.append(
                {
                    "table_name": table,
                    "deleted_rows": cursor.rowcount,
                }
            )

    return pd.DataFrame(delete_log)


def main():
    db = CyclingDatabase(db_path="../../data/databases/cycling_production.db")
    conn = db.get_connection()

    print("==================================================================")
    print(" 🧹 DELETE CHECK: Fahrer aus DB löschen")
    print("==================================================================")

    target_urls, df_found = resolve_rider_urls(conn)

    print("\n------------------------------------------------------------------")
    print(" 🔎 Gefundene Fahrer in r_master")
    print("------------------------------------------------------------------")

    if df_found.empty:
        print("Keine Treffer in r_master gefunden.")
    else:
        print(df_found.to_string(index=False))

    print("\n------------------------------------------------------------------")
    print(" 🎯 Ziel-rider_urls")
    print("------------------------------------------------------------------")
    for url in target_urls:
        print(f"- {url}")

    print("\n------------------------------------------------------------------")
    print(" 📊 Vorschau: Zeilen, die gelöscht würden")
    print("------------------------------------------------------------------")

    df_preview = preview_counts(conn, target_urls)

    if df_preview.empty:
        print("Keine passenden Zeilen in Tabellen mit rider_url gefunden.")
    else:
        print(df_preview.to_string(index=False))
        print(f"\n➔ Summe Zeilen: {df_preview['rows_to_delete'].sum()}")

    if DRY_RUN:
        print("\n==================================================================")
        print(" 🟡 DRY_RUN = True")
        print(" Es wurde noch nichts gelöscht.")
        print(" Wenn die Vorschau passt: DRY_RUN = False setzen und erneut ausführen.")
        print("==================================================================")
        conn.close()
        return

    print("\n==================================================================")
    print(" 🛠️ LÖSCHE FAHRER AUS ALLEN TABELLEN MIT rider_url")
    print("==================================================================")

    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN")

        df_deleted = delete_riders(conn, target_urls)

        conn.commit()

        if df_deleted.empty:
            print("Keine Zeilen gelöscht.")
        else:
            print(df_deleted.to_string(index=False))
            print(f"\n➔ Gelöschte Zeilen gesamt: {df_deleted['deleted_rows'].sum()}")

    except Exception as e:
        conn.rollback()
        print(f"❌ Fehler beim Löschen. Rollback durchgeführt: {e}")

    finally:
        conn.close()

    print("\n==================================================================")
    print(" ✅ FERTIG")
    print("==================================================================")


if __name__ == "__main__":
    main()
