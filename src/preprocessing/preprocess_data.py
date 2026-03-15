"""
Data Preprocessing Script
Cleans, transforms, and prepares raw data for the pipeline.
Saves preprocessed data to data/preprocessed/

"""

import json
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import RAW_DATA_DIR, PREPROCESSED_DATA_DIR
from src.preprocessing.invalid_record_detection import (
    get_all_critical_fields_invalid_mask,
)
from src.preprocessing.language_detection import detect_language_distribution


def get_language_check(df, field='Description', mode='sample', sample_size=500):
    """Reusable language check for preprocessing decisions."""
    if field not in df.columns:
        return None
    return detect_language_distribution(
        df[field],
        mode=mode,
        sample_size=sample_size,
    )


def remove_records_with_all_critical_fields_invalid(df):
    """
    Detect and remove job-posting records where every critical field is invalid.

    A record is considered invalid only when all existing critical fields are invalid
    at the same time. This keeps rows that still contain useful information in at
    least one important field.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, list[str]]:
            - cleaned_df: DataFrame after removing fully invalid records
            - invalid_records: DataFrame containing only the removed records
            - checked_fields: critical fields that were actually present and checked
    """
    invalid_mask, checked_fields = get_all_critical_fields_invalid_mask(df)

    # If none of the expected critical fields exist, there is nothing to filter.
    if invalid_mask is None:
        return df.copy(), df.iloc[0:0].copy(), checked_fields

    # Keep a separate copy of removed rows for reporting and debugging.
    invalid_records = df[invalid_mask].copy()

    # Keep only rows that still have at least one usable critical field.
    cleaned_df = df[~invalid_mask].copy()

    return cleaned_df, invalid_records, checked_fields


def preprocess_ecsf(data):
    """
    Preprocess ECSF data
    
    TODO: Implement based on exploration findings
    - Standardize field names
    - Extract relevant fields
    """
    print("🔧 Preprocessing ECSF data...")
    
    pass


def preprocess_job_postings(data):
    """
    Preprocess Job Postings data
    - Clean text fields
    - Extract skills/requirements text
    - Handle missing values
    - Standardize location data
    - Remove records with all critical fields invalid
    """
    print("🔧 Preprocessing Job Postings data...")

    df = pd.DataFrame(data)
    print(f"  Loaded {len(df)} job posting records")

    # Step 1: remove rows where every critical field is invalid.
    cleaned_df, invalid_records, checked_fields = remove_records_with_all_critical_fields_invalid(df)

    if checked_fields:
        print(f"  Critical fields checked: {checked_fields}")
        print(f"  Invalid records removed: {len(invalid_records)}")
        print(f"  Remaining records: {len(cleaned_df)}")
    else:
        print("  No critical fields found for invalid-record detection")

    return cleaned_df.to_dict('records')


def save_preprocessed_data(data, filename):
    """Save preprocessed data to JSON"""
    output_path = PREPROCESSED_DATA_DIR / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved: {output_path}")


def main():
    """Main preprocessing function"""
    print("⚙️  Starting Data Preprocessing...\n")
    
    # Load raw data
    # ecsf_file = RAW_DATA_DIR / 'ecsf.json'
    job_postings_file = RAW_DATA_DIR / 'job_postings.json'
    
    if not job_postings_file.exists():
        print("❌ Raw data files not found. Run explore_data.py first.")
        return
    
    # Load
    # with open(ecsf_file, 'r', encoding='utf-8') as f:
    #     ecsf_data = json.load(f)
    
    with open(job_postings_file, 'r', encoding='utf-8') as f:
        job_data = json.load(f)
    
    # ecsf_preprocessed = preprocess_ecsf(ecsf_data)
    job_preprocessed = preprocess_job_postings(job_data)

    # Save preprocessed data
    # save_preprocessed_data(ecsf_preprocessed, 'ecsf_preprocessed.json')
    save_preprocessed_data(job_preprocessed, 'job_postings_preprocessed.json')


if __name__ == '__main__':
    main()
