"""
Shared utilities for language detection and invalid-text filtering.
"""

from collections import Counter

import pandas as pd
from src.utils.sampling import sample_collection

INVALID_CONTENT_VALUES = {
    "",
    " ",
    "N/A",
    "None",
    "none",
    "n/a",
    "null",
    "NULL",
}

try:
    from langdetect import detect, LangDetectException

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    detect = None

    class LangDetectException(Exception):
        """Fallback exception when langdetect is unavailable."""


def invalid_content_mask(series, invalid_values=None):
    """Return a boolean mask indicating invalid textual content in a Series."""
    values = invalid_values or INVALID_CONTENT_VALUES
    cleaned = series.fillna("").astype(str).str.strip()
    return cleaned.isin(values)


def get_valid_texts(series, invalid_values=None):
    """Return only valid text values from a Series."""
    return series[~invalid_content_mask(series, invalid_values)].astype(str)


def detect_language_distribution(series, mode="sample", sample_size=500, invalid_values=None):
    """
    Detect language distribution in a text Series.

    Returns a dict with:
    - available: whether langdetect is installed
    - sampled_records: number of analyzed rows
    - language_counts: Counter with language frequencies
    - english_percentage: percentage of english among sampled records
    - mode: sample or full
    """
    if not LANGDETECT_AVAILABLE:
        return {
            "available": False,
            "sampled_records": 0,
            "language_counts": Counter(),
            "english_percentage": 0.0,
            "mode": mode,
        }

    valid_texts = get_valid_texts(series, invalid_values)

    if mode not in {"sample", "full"}:
        raise ValueError("mode must be 'sample' or 'full'")

    texts = sample_collection(valid_texts, mode=mode, sample_size=sample_size)

    detected = []
    for text in texts:
        try:
            detected.append(detect(text))
        except LangDetectException:
            detected.append("unknown")

    language_counts = Counter(detected)
    sampled_records = len(detected)
    english_percentage = 0.0
    if sampled_records > 0:
        english_percentage = (language_counts.get("en", 0) / sampled_records) * 100

    return {
        "available": True,
        "sampled_records": sampled_records,
        "language_counts": language_counts,
        "english_percentage": english_percentage,
        "mode": mode,
    }
