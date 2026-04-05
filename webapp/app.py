"""Flask web application for showing results."""

from urllib.parse import urlencode

from flask import Flask, render_template, request

from src.utils.database import db

app = Flask(__name__)


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
    top_k = int(request.args.get("top_k", 5))
    min_similarity = float(request.args.get("min_similarity", 0.50))
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
      AND similarity >= %(min_similarity)s
      AND rank <= %(top_k)s
    """
    params = {
        "model_name": model_name,
        "min_similarity": min_similarity,
        "top_k": top_k,
    }

    if skill_query:
        base_query += " AND normalized_skill ILIKE %(skill_query)s"
        params["skill_query"] = f"%{skill_query}%"

    count_query = f"SELECT COUNT(*) AS total FROM ({base_query}) AS filtered"
    query = (
        base_query
        + " ORDER BY normalized_skill, similarity DESC LIMIT %(limit)s OFFSET %(offset)s"
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
        "min_similarity": min_similarity,
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
        min_similarity=min_similarity,
        skill_query=skill_query,
        page=page,
        page_size=page_size,
        total_rows=total_rows,
        total_pages=total_pages,
        pagination=pagination,
    )


@app.route('/health')
def health():
    """Health check."""
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
