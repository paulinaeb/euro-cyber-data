"""Evaluate skill-to-ECSF mapping quality metrics."""

import argparse
from typing import List

from src.utils.database import db, ensure_database_exists


def parse_thresholds(value: str) -> List[float]:
    thresholds = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        thresholds.append(float(part))
    return thresholds


def fetch_total_skills() -> int:
    result = db.execute_query("SELECT COUNT(*) AS total FROM skill_dim")
    return result[0]["total"] if result else 0


def fetch_best_similarities(model_name: str):
    query = """
    SELECT skill_id, MAX(similarity) AS best_similarity
    FROM skill_ecsf_similarity
    WHERE model_name = %(model_name)s
    GROUP BY skill_id
    """
    return db.execute_query(query, {"model_name": model_name})


def fetch_mean_similarity(model_name: str) -> float:
    query = """
    SELECT AVG(best_similarity) AS mean_similarity
    FROM (
        SELECT skill_id, MAX(similarity) AS best_similarity
        FROM skill_ecsf_similarity
        WHERE model_name = %(model_name)s
        GROUP BY skill_id
    ) AS best
    """
    result = db.execute_query(query, {"model_name": model_name})
    value = result[0]["mean_similarity"] if result else None
    return float(value) if value is not None else 0.0


def main():
    created = ensure_database_exists()
    if created:
        print("Created database")
    else:
        print("Database already exists")

    parser = argparse.ArgumentParser(description="Evaluate skill-to-ECSF mapping metrics")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name used in embeddings",
    )
    parser.add_argument(
        "--thresholds",
        default="0.55,0.65,0.75",
        help="Comma-separated similarity thresholds",
    )
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    total_skills = fetch_total_skills()
    best_rows = fetch_best_similarities(args.model_name)
    skills_with_match = len(best_rows)
    mean_similarity = fetch_mean_similarity(args.model_name)

    print("Evaluation summary")
    print(f"Model: {args.model_name}")
    print(f"Total skills: {total_skills}")
    print(f"Skills with >=1 match: {skills_with_match}")
    print(f"Semantic Similarity Mean (mu): {mean_similarity:.4f}")

    if total_skills == 0:
        print("No skills available for coverage calculation")
        return

    for threshold in thresholds:
        covered = sum(1 for row in best_rows if row["best_similarity"] >= threshold)
        coverage = covered / total_skills
        print(f"Coverage@{threshold:.2f}: {coverage:.4f} ({covered}/{total_skills})")


if __name__ == "__main__":
    main()
