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


def preprocess_ecsf(data):
    """
    Preprocess ECSF data
    
    TODO: Implement based on exploration findings
    - Remove duplicates
    - Standardize field names
    - Handle missing values
    - Extract relevant fields
    """
    print("🔧 Preprocessing ECSF data...")
    
    # Placeholder - implement after exploration
    pass


def preprocess_job_postings(data):
    """
    Preprocess Job Postings data
    
    TODO: Implement based on exploration findings
    - Remove duplicates
    - Clean text fields
    - Extract skills/requirements text
    - Handle missing values
    - Standardize location data
    """
    print("🔧 Preprocessing Job Postings data...")
    
    # Placeholder - implement after exploration
    pass


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
    ecsf_file = RAW_DATA_DIR / 'ecsf.json'
    job_postings_file = RAW_DATA_DIR / 'job_postings.json'
    
    if not ecsf_file.exists() or not job_postings_file.exists():
        print("❌ Raw data files not found. Run explore_data.py first.")
        return
    
    # Load
    with open(ecsf_file, 'r', encoding='utf-8') as f:
        ecsf_data = json.load(f)
    
    with open(job_postings_file, 'r', encoding='utf-8') as f:
        job_data = json.load(f)
    
    
    # Save preprocessed data
    # save_preprocessed_data(ecsf_preprocessed, 'ecsf_preprocessed.json')
    # save_preprocessed_data(job_preprocessed, 'job_postings_preprocessed.json')


if __name__ == '__main__':
    main()
