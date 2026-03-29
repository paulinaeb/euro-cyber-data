"""
Load ECSF data into PostgreSQL.

Creates the database and ECSF tables if needed, then loads data from the
raw JSON file only when tables are empty.
"""

import argparse
import json
from pathlib import Path

from src.utils.config import PREPROCESSED_DIR
from src.utils.database import db, ensure_database_exists


ECSF_FILE = PREPROCESSED_DIR / "ecsf_preprocessed.json"
WORK_ROLE_TABLE = "ecsf_work_role"
TKS_TABLE = "ecsf_tks"
REL_TABLE = "ecsf_relationship"


def load_ecsf_data(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_tables_exist():
    work_role_sql = f"""
    CREATE TABLE IF NOT EXISTS {WORK_ROLE_TABLE} (
        id INTEGER PRIMARY KEY,
        title TEXT,
        alternative_titles TEXT[]
    )
    """
    tks_sql = f"""
    CREATE TABLE IF NOT EXISTS {TKS_TABLE} (
        id TEXT PRIMARY KEY,
        type TEXT,
        description TEXT
    )
    """
    rel_sql = f"""
    CREATE TABLE IF NOT EXISTS {REL_TABLE} (
        work_role_id INTEGER REFERENCES {WORK_ROLE_TABLE}(id),
        tks_id TEXT REFERENCES {TKS_TABLE}(id),
        PRIMARY KEY (work_role_id, tks_id)
    )
    """
    db.execute_query(work_role_sql, fetch=False)
    db.execute_query(tks_sql, fetch=False)
    db.execute_query(rel_sql, fetch=False)


def table_has_rows(table_name):
    result = db.execute_query(f"SELECT COUNT(*) AS count FROM {table_name}")
    return result[0]["count"] > 0


def insert_work_roles(records, batch_size=500):
    insert_sql = f"""
    INSERT INTO {WORK_ROLE_TABLE} (
        id,
        title,
        alternative_titles
    ) VALUES (
        %(id)s,
        %(title)s,
        %(alternative_titles)s
    )
    """
    db.execute_many(insert_sql, records, batch_size=batch_size)


def insert_tks(records, batch_size=1000):
    insert_sql = f"""
    INSERT INTO {TKS_TABLE} (
        id,
        type,
        description
    ) VALUES (
        %(id)s,
        %(type)s,
        %(description)s
    )
    """
    db.execute_many(insert_sql, records, batch_size=batch_size)


def insert_relationships(records, batch_size=2000):
    insert_sql = f"""
    INSERT INTO {REL_TABLE} (
        work_role_id,
        tks_id
    ) VALUES (
        %(work_role_id)s,
        %(tks_id)s
    )
    """
    db.execute_many(insert_sql, records, batch_size=batch_size)


def map_work_roles(records):
    mapped = []
    for record in records:
        mapped.append(
            {
                "id": record.get("id"),
                "title": record.get("title"),
                "alternative_titles": record.get("alternative_title(s)"),
            }
        )
    return mapped


def map_tks(records):
    mapped = []
    for record in records:
        mapped.append(
            {
                "id": record.get("id"),
                "type": record.get("type"),
                "description": record.get("description"),
            }
        )
    return mapped


def map_relationships(records):
    mapped = []
    for record in records:
        mapped.append(
            {
                "work_role_id": record.get("work_role_id"),
                "tks_id": record.get("tks_id"),
            }
        )
    return mapped


def main():
    parser = argparse.ArgumentParser(description="Create ECSF tables and load data if empty.")
    parser.add_argument(
        "--input-file",
        default=str(ECSF_FILE),
        help="Path to the ECSF JSON file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    created = ensure_database_exists()
    if created:
        print("Created database")
    else:
        print("Database already exists")

    ensure_tables_exist()

    if table_has_rows(WORK_ROLE_TABLE) or table_has_rows(TKS_TABLE) or table_has_rows(REL_TABLE):
        print("ECSF tables already have data; skipping load")
        return

    data = load_ecsf_data(input_path)
    work_roles = map_work_roles(data.get("work_role", []))
    tks = map_tks(data.get("tks", []))
    relationships = map_relationships(data.get("relationship", []))

    if work_roles:
        insert_work_roles(work_roles)
        print(f"Inserted {len(work_roles)} work roles")

    if tks:
        insert_tks(tks)
        print(f"Inserted {len(tks)} tks records")

    if relationships:
        insert_relationships(relationships)
        print(f"Inserted {len(relationships)} relationships")


if __name__ == "__main__":
    main()
