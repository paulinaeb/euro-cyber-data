"""Flask web application for showing results."""

from urllib.parse import urlencode

from flask import Flask, render_template, request

from src.utils.database import db

app = Flask(__name__)

SIMILARITY_OPTIONS = (0.55, 0.65, 0.75)

EU_COUNTRIES = {
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czechia",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
}


@app.route('/')
def index():
    """Home page."""
    return render_template("index.html")


@app.route("/mapping")
def mapping():
    """Display interpretable skill to ECSF mappings."""
    model_name = request.args.get(
        "model_name", "sentence-transformers/all-MiniLM-L6-v2"
    )
    top_k = int(request.args.get("top_k", 3))
    similarity_at = float(request.args.get("similarity_at", SIMILARITY_OPTIONS[0]))
    if similarity_at not in SIMILARITY_OPTIONS:
        similarity_at = SIMILARITY_OPTIONS[0]
    if similarity_at == 0.75:
        similarity_upper = 1.0
    else:
        similarity_upper = min(similarity_at + 0.10, 1.0)
    skill_query = request.args.get("skill", "").strip()
    page = max(int(request.args.get("page", 1)), 1)
    page_size = int(request.args.get("page_size", 50))
    page_size = max(min(page_size, 200), 10)
    offset = (page - 1) * page_size

    base_query = """
    SELECT
        skill_id,
        normalized_skill,
        tks_id,
        tks_description,
        similarity,
        rank
    FROM vw_skill_ecsf_mapping
    WHERE model_name = %(model_name)s
            AND similarity >= %(similarity_at)s
            AND (
                similarity < %(similarity_upper)s
                OR (%(similarity_upper)s = 1.0 AND similarity = 1.0)
            )
      AND rank <= %(top_k)s
    """
    params = {
        "model_name": model_name,
                "similarity_at": similarity_at,
                "similarity_upper": similarity_upper,
        "top_k": top_k,
    }

    if skill_query:
        base_query += " AND normalized_skill ILIKE %(skill_query)s"
        params["skill_query"] = f"%{skill_query}%"

    count_query = f"SELECT COUNT(*) AS total FROM ({base_query}) AS filtered"
    query = (
        base_query
        + " ORDER BY similarity ASC, normalized_skill ASC LIMIT %(limit)s OFFSET %(offset)s"
    )
    params["limit"] = page_size
    params["offset"] = offset

    error = None
    rows = []
    total_rows = 0
    try:
        count_result = db.execute_query(count_query, params)
        total_rows = count_result[0]["total"] if count_result else 0
        rows = db.execute_query(query, params)
    except Exception as exc:
        error = str(exc)

    total_pages = max((total_rows + page_size - 1) // page_size, 1)
    page = min(page, total_pages)

    query_params = {
        "model_name": model_name,
        "top_k": top_k,
        "similarity_at": similarity_at,
        "skill": skill_query,
        "page_size": page_size,
    }

    def build_page_url(target_page: int) -> str:
        params = query_params.copy()
        params["page"] = target_page
        return f"/mapping?{urlencode(params)}"

    pagination = {
        "first": build_page_url(1) if page > 1 else None,
        "prev": build_page_url(page - 1) if page > 1 else None,
        "next": build_page_url(page + 1) if page < total_pages else None,
        "last": build_page_url(total_pages) if page < total_pages else None,
    }

    return render_template(
        "mapping.html",
        rows=rows,
        error=error,
        model_name=model_name,
        top_k=top_k,
        similarity_at=similarity_at,
        similarity_upper=similarity_upper,
        similarity_options=SIMILARITY_OPTIONS,
        skill_query=skill_query,
        page=page,
        page_size=page_size,
        total_rows=total_rows,
        total_pages=total_pages,
        pagination=pagination,
    )


@app.route("/frequency")
def frequency():
    """Show normalized skill frequency distribution."""
    top_n = int(request.args.get("top_n", 30))
    top_n = max(min(top_n, 200), 5)

    count_query = """
    SELECT s.normalized_skill, COUNT(DISTINCT js.job_posting_id) AS occurrences
    FROM job_skill js
    INNER JOIN skill_dim s ON s.id = js.skill_id
    GROUP BY s.normalized_skill
    ORDER BY occurrences DESC
    LIMIT %(limit)s
    """
    total_query = "SELECT COUNT(*) AS total FROM job_postings"

    error = None
    rows = []
    total_count = 0
    try:
        total_result = db.execute_query(total_query)
        total_count = total_result[0]["total"] if total_result else 0
        rows = db.execute_query(count_query, {"limit": top_n})
    except Exception as exc:
        error = str(exc)

    enriched = []
    for row in rows:
        occurrences = row["occurrences"]
        frequency = (occurrences / total_count) if total_count else 0.0
        frequency_pct = frequency * 100
        enriched.append(
            {
                "normalized_skill": row["normalized_skill"],
                "occurrences": occurrences,
                "frequency": frequency,
                "frequency_pct": frequency_pct,
            }
        )

    labels = [row["normalized_skill"] for row in enriched]
    values = [row["occurrences"] for row in enriched]
    label_suffix = [round(row["frequency_pct"], 2) for row in enriched]

    return render_template(
        "frequency.html",
        rows=enriched,
        error=error,
        total_count=total_count,
        top_n=top_n,
        labels=labels,
        values=values,
        label_suffix=label_suffix,
        total_postings=total_count,
    )


@app.route("/geo-summary")
def geo_summary():
    """Show EU skill demand summary with optional non-EU inclusion."""
    include_non_eu = request.args.get("include_non_eu", "false").lower() == "true"
    top_n = int(request.args.get("top_n", 5))
    top_n = max(min(top_n, 10), 3)

    base_query = """
    SELECT jp.country, s.normalized_skill, COUNT(DISTINCT js.job_posting_id) AS occurrences
    FROM job_postings jp
    INNER JOIN job_skill js ON js.job_posting_id = jp.id
    INNER JOIN skill_dim s ON s.id = js.skill_id
    WHERE jp.country IS NOT NULL
    """
    postings_query = """
    SELECT country, COUNT(DISTINCT id) AS total_postings
    FROM job_postings
    WHERE country IS NOT NULL
    GROUP BY country
    """
    params = {}
    if not include_non_eu:
        base_query += " AND jp.country = ANY(%(countries)s)"
        postings_query = postings_query.replace(
            "WHERE country IS NOT NULL",
            "WHERE country IS NOT NULL AND country = ANY(%(countries)s)",
        )
        params["countries"] = sorted(EU_COUNTRIES)

    base_query += " GROUP BY jp.country, s.normalized_skill"

    error = None
    rows = []
    try:
        rows = db.execute_query(base_query, params)
        posting_rows = db.execute_query(postings_query, params)
    except Exception as exc:
        error = str(exc)
        posting_rows = []

    postings_lookup = {
        row["country"]: row["total_postings"] for row in posting_rows
    }

    countries = {}
    for row in rows:
        country = row["country"]
        occurrences = row["occurrences"]
        skill = row["normalized_skill"]
        entry = countries.setdefault(
            country,
            {
                "country": country,
                "total_postings": postings_lookup.get(country, 0),
                "skills": {},
            },
        )
        entry["skills"][skill] = occurrences

    country_summaries = []
    for entry in countries.values():
        total_postings = entry["total_postings"]
        sorted_skills = sorted(
            entry["skills"].items(), key=lambda item: item[1], reverse=True
        )
        top_skills = []
        for skill, count in sorted_skills[:top_n]:
            frequency = count / total_postings if total_postings else 0.0
            frequency_pct = frequency * 100
            top_skills.append(
                {
                    "skill": skill,
                    "count": count,
                    "frequency": round(frequency, 4),
                    "frequency_pct": round(frequency_pct, 2),
                }
            )

        top_skill = top_skills[0] if top_skills else None
        country_summaries.append(
            {
                "country": entry["country"],
                "total_postings": total_postings,
                "top_skills": top_skills,
                "top_skill": top_skill,
            }
        )

    country_summaries = sorted(
        country_summaries, key=lambda item: item["total_postings"], reverse=True
    )

    return render_template(
        "geo_summary.html",
        error=error,
        include_non_eu=include_non_eu,
        top_n=top_n,
        summaries=country_summaries,
    )


@app.route('/health')
def health():
    """Health check."""
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
