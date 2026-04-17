"""Configuration Module"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"

# Database
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "database": os.getenv("POSTGRES_DB", "euro_cyber_db"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
}

# Data files
JOB_POSTINGS_FILE = os.getenv("JOB_POSTINGS_FILE", "job_postings.json")
ECSF_FILE = os.getenv("ECSF_FILE", "ecsf.json")

# Pipeline
SBERT_MODEL = os.getenv("SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.55))

# Ensure directories exist
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_data_path(filename: str, data_type: str = "raw") -> Path:
    """Get full path for a data file"""
    if data_type == "raw":
        return RAW_DATA_DIR / filename
    elif data_type == "preprocessed":
        return PREPROCESSED_DIR / filename
    else:
        raise ValueError(f"Invalid data_type: {data_type}")
