"""
Create extraction-related tables for skill mapping and embeddings.
"""

from src.utils.database import db, ensure_database_exists

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


def main():
    created = ensure_database_exists()
    if created:
        print("Created database")
    else:
        print("Database already exists")

    ensure_tables_exist()
    print("Extraction tables ensured")


if __name__ == "__main__":
    main()
