"""Compute cosine similarity between skill and ECSF embeddings."""

import argparse
from typing import List, Dict

import numpy as np

from src.utils.database import db, ensure_database_exists

SKILL_EMBEDDING_TABLE = "skill_embedding"
ECSF_EMBEDDING_TABLE = "ecsf_tks_embedding"
SIMILARITY_TABLE = "skill_ecsf_similarity"


def fetch_embeddings(table: str, id_col: str, model_name: str) -> List[Dict]:
    query = f"""
    SELECT {id_col}, embedding
    FROM {table}
    WHERE model_name = %(model_name)s
    ORDER BY {id_col}
    """
    return db.execute_query(query, {"model_name": model_name})


def insert_similarity(rows: List[Dict]):
    if not rows:
        return
    insert_sql = f"""
    INSERT INTO {SIMILARITY_TABLE} (
        skill_id,
        tks_id,
        model_name,
        similarity
    ) VALUES (
        %(skill_id)s,
        %(tks_id)s,
        %(model_name)s,
        %(similarity)s
    )
    ON CONFLICT (skill_id, tks_id, model_name) DO NOTHING
    """
    db.execute_many(insert_sql, rows, batch_size=2000)


def compute_similarity(model_name: str, top_k: int, min_similarity: float, batch_size: int):
    skill_rows = fetch_embeddings(SKILL_EMBEDDING_TABLE, "skill_id", model_name)
    ecsf_rows = fetch_embeddings(ECSF_EMBEDDING_TABLE, "tks_id", model_name)

    if not skill_rows:
        print("No skill embeddings found")
        return
    if not ecsf_rows:
        print("No ECSF embeddings found")
        return

    skill_ids = [row["skill_id"] for row in skill_rows]
    ecsf_ids = [row["tks_id"] for row in ecsf_rows]

    skill_matrix = np.asarray([row["embedding"] for row in skill_rows], dtype=np.float32)
    ecsf_matrix = np.asarray([row["embedding"] for row in ecsf_rows], dtype=np.float32)

    if skill_matrix.ndim != 2 or ecsf_matrix.ndim != 2:
        print("Invalid embedding shapes")
        return

    ecsf_matrix_t = ecsf_matrix.T

    total_inserted = 0
    for start in range(0, len(skill_ids), batch_size):
        end = start + batch_size
        skill_chunk = skill_matrix[start:end]
        sims = np.matmul(skill_chunk, ecsf_matrix_t)

        k = min(top_k, sims.shape[1])
        top_idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]

        payload = []
        for i, idxs in enumerate(top_idx):
            row_index = start + i
            skill_id = skill_ids[row_index]
            for idx in idxs:
                similarity = float(sims[i, idx])
                if similarity < min_similarity:
                    continue
                payload.append(
                    {
                        "skill_id": skill_id,
                        "tks_id": ecsf_ids[idx],
                        "model_name": model_name,
                        "similarity": similarity,
                    }
                )

        insert_similarity(payload)
        total_inserted += len(payload)

    print(f"Inserted {total_inserted} skill-ecsf similarity rows")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute skill to ECSF similarity")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name to filter embeddings",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main():
    created = ensure_database_exists()
    if created:
        print("Created database")
    else:
        print("Database already exists")

    args = parse_args()
    compute_similarity(args.model_name, args.top_k, args.min_similarity, args.batch_size)


if __name__ == "__main__":
    main()
