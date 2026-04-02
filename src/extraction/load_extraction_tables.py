"""
Populate extraction tables from existing database data.
"""

from src.utils.database import db, ensure_database_exists
from src.extraction.skill_normalizer import (
    split_skill_field,
    normalize_skill,
    is_noise_skill,
)

SKILL_DIM_TABLE = "skill_dim"
JOB_SKILL_TABLE = "job_skill"
ECSF_TKS_TEXT_TABLE = "ecsf_tks_text"


def ensure_tables_exist():
    skill_dim_sql = f"""
    CREATE TABLE IF NOT EXISTS {SKILL_DIM_TABLE} (
        id SERIAL PRIMARY KEY,
        normalized_skill TEXT UNIQUE
    )
    """

    job_skill_sql = f"""
    CREATE TABLE IF NOT EXISTS {JOB_SKILL_TABLE} (
        job_posting_id INTEGER REFERENCES job_postings(id),
        skill_id INTEGER REFERENCES {SKILL_DIM_TABLE}(id),
        raw_skill TEXT,
        PRIMARY KEY (job_posting_id, skill_id)
    )
    """

    ecsf_tks_text_sql = f"""
    CREATE TABLE IF NOT EXISTS {ECSF_TKS_TEXT_TABLE} (
        id SERIAL PRIMARY KEY,
        tks_id TEXT REFERENCES ecsf_tks(id),
        embedding_text TEXT,
        UNIQUE (tks_id)
    )
    """

    db.execute_query(skill_dim_sql, fetch=False)
    db.execute_query(job_skill_sql, fetch=False)
    db.execute_query(ecsf_tks_text_sql, fetch=False)


def table_count(table_name):
    result = db.execute_query(f"SELECT COUNT(*) AS count FROM {table_name}")
    return result[0]["count"]


def load_skill_dim():
    rows = db.execute_query(
        "SELECT id, skill FROM job_postings WHERE skill IS NOT NULL"
    )
    skill_map = set()
    for row in rows:
        for raw_skill in split_skill_field(row["skill"]):
            if is_noise_skill(raw_skill):
                continue
            normalized = normalize_skill(raw_skill)
            if not normalized.normalized_skill:
                continue
            skill_map.add(normalized.normalized_skill)

    if not skill_map:
        print(f"No skills found for {SKILL_DIM_TABLE}")
        return

    insert_sql = f"""
    INSERT INTO {SKILL_DIM_TABLE} (
        normalized_skill
    ) VALUES (
        %(normalized_skill)s
    )
    ON CONFLICT (normalized_skill)
    DO NOTHING
    """

    before_count = table_count(SKILL_DIM_TABLE)
    payload = [
        {"normalized_skill": normalized_skill}
        for normalized_skill in skill_map
    ]
    db.execute_many(insert_sql, payload, batch_size=1000)
    after_count = table_count(SKILL_DIM_TABLE)
    print(f"Inserted {after_count - before_count} skills into {SKILL_DIM_TABLE}")


def load_job_skills():
    skill_rows = db.execute_query(
        f"SELECT id, normalized_skill FROM {SKILL_DIM_TABLE}"
    )
    skill_lookup = {row["normalized_skill"]: row["id"] for row in skill_rows}

    rows = db.execute_query(
        "SELECT id, skill FROM job_postings WHERE skill IS NOT NULL"
    )

    payload = []
    for row in rows:
        job_posting_id = row["id"]
        for raw_skill in split_skill_field(row["skill"]):
            if is_noise_skill(raw_skill):
                continue
            normalized = normalize_skill(raw_skill)
            skill_id = skill_lookup.get(normalized.normalized_skill)
            if skill_id is None:
                continue
            payload.append(
                {
                    "job_posting_id": job_posting_id,
                    "skill_id": skill_id,
                    "raw_skill": normalized.raw_skill,
                }
            )

    if not payload:
        print("No job skill links to insert")
        return

    insert_sql = f"""
    INSERT INTO {JOB_SKILL_TABLE} (
        job_posting_id,
        skill_id,
        raw_skill
    ) VALUES (
        %(job_posting_id)s,
        %(skill_id)s,
        %(raw_skill)s
    )
    ON CONFLICT (job_posting_id, skill_id) DO NOTHING
    """

    before_count = table_count(JOB_SKILL_TABLE)
    db.execute_many(insert_sql, payload, batch_size=2000)
    after_count = table_count(JOB_SKILL_TABLE)
    print(f"Inserted {after_count - before_count} job skill links")


def load_ecsf_tks_text():
    insert_sql = f"""
    INSERT INTO {ECSF_TKS_TEXT_TABLE} (
        tks_id,
        embedding_text
    )
    SELECT
        id AS tks_id,
        description AS embedding_text
    FROM ecsf_tks
    WHERE type IN ('Knowledge', 'Skill')
      AND description IS NOT NULL
    ON CONFLICT (tks_id) DO NOTHING
    """

    before_count = table_count(ECSF_TKS_TEXT_TABLE)
    db.execute_query(insert_sql, fetch=False)
    after_count = table_count(ECSF_TKS_TEXT_TABLE)
    print(f"Inserted {after_count - before_count} ECSF tks texts")


def main():
    created = ensure_database_exists()
    if created:
        print("Created database")
    else:
        print("Database already exists")

    ensure_tables_exist()
    load_skill_dim()
    load_job_skills()
    load_ecsf_tks_text()


if __name__ == "__main__":
    main()
