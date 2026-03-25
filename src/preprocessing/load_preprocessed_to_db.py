"""
Load preprocessed job postings into PostgreSQL.

Creates the database and tables if needed, then loads data from the
preprocessed JSON file only when the table is empty.
"""

import argparse
import json
from pathlib import Path

from src.utils.cli_args import add_sample_mode_arguments, is_valid_sample_size
from src.utils.config import PREPROCESSED_DIR
from src.utils.database import db, ensure_database_exists
from src.utils.sampling import sample_collection


TABLE_NAME = "job_postings"


def load_preprocessed_data(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_table_exists():
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        title TEXT,
        description TEXT,
        company TEXT,
        primary_description TEXT,
        work_modality TEXT,
        location TEXT,
        country TEXT,
        skill TEXT,
        created_at TIMESTAMPTZ,
        scraped_at TIMESTAMPTZ
    )
    """
    db.execute_query(create_sql, fetch=False)


def table_has_rows():
    result = db.execute_query(f"SELECT COUNT(*) AS count FROM {TABLE_NAME}")
    return result[0]["count"] > 0


def insert_rows(rows, batch_size=1000):
    insert_sql = f"""
    INSERT INTO {TABLE_NAME} (
        title,
        description,
        company,
        primary_description,
        work_modality,
        location,
        country,
        skill,
        created_at,
        scraped_at
    ) VALUES (
        %(title)s,
        %(description)s,
        %(company)s,
        %(primary_description)s,
        %(work_modality)s,
        %(location)s,
        %(country)s,
        %(skill)s,
        %(created_at)s,
        %(scraped_at)s
    )
    """
    db.execute_many(insert_sql, rows, batch_size=batch_size)


def map_record(record):
    return {
        "title": record.get("Title"),
        "description": record.get("Description"),
        "company": record.get("Company"),
        "primary_description": record.get("Primary Description"),
        "work_modality": record.get("Work Modality"),
        "location": record.get("Location"),
        "country": record.get("Country"),
        "skill": record.get("Skill"),
        "created_at": record.get("Created At"),
        "scraped_at": record.get("Scraped At"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create database tables and load preprocessed job postings if empty."
    )
    add_sample_mode_arguments(
        parser,
        mode_flag="--run-mode",
        mode_dest="run_mode",
        default_mode="full",
        mode_help="Execution mode: 'sample' for quick iteration or 'full' for complete dataset (default: full).",
        sample_size_default=1000,
        sample_size_help="Number of records used when --run-mode=sample (default: 1000).",
    )
    parser.add_argument(
        "--input-file",
        default=str(PREPROCESSED_DIR / "job_postings_preprocessed.json"),
        help="Path to the preprocessed input JSON.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for inserts (default: 1000).",
    )
    args = parser.parse_args()

    if not is_valid_sample_size(args.sample_size):
        print("sample-size must be greater than 0")
        return

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    created = ensure_database_exists()
    if created:
        print("Created database")
    else:
        print("Database already exists")

    ensure_table_exists()

    if table_has_rows():
        print("Table already has data; skipping load")
        return

    data = load_preprocessed_data(input_path)
    data = sample_collection(data, mode=args.run_mode, sample_size=args.sample_size)
    rows = [map_record(record) for record in data]

    if not rows:
        print("No rows to insert")
        return

    insert_rows(rows, batch_size=args.batch_size)
    print(f"Inserted {len(rows)} rows into {TABLE_NAME}")


if __name__ == "__main__":
    main()
