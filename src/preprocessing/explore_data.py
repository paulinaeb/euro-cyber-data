"""
Data Exploration
Analyzes raw JSON files (ECSF and Job Postings) to understand structure,
quality, and characteristics before preprocessing.
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import RAW_DATA_DIR

# Language detection
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️  Warning: langdetect not installed. Language detection will be skipped.")


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def explore_ecsf(filepath):
    """Explore ECSF data structure"""
    print("=" * 80)
    print("ECSF DATA EXPLORATION")
    print("=" * 80)
    
    data = load_json(filepath)
    
    # Check if it's a list or dict
    print(f"\n📊 Data type: {type(data).__name__}")
    
    if isinstance(data, list):
        print(f"📊 Total records: {len(data)}")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(data)
        
        print(f"\n📋 Columns ({len(df.columns)}):")
        for col in df.columns:
            print(f"  - {col}")
        
        print(f"\n📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        print("\n🔍 Column Data Types:")
        print(df.dtypes)
        
        print("\n⚠️  Missing Values:")
        missing = df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "  No missing values")
        
        print("\n🔄 Duplicate Records:")
        duplicates = df.duplicated().sum()
        print(f"  {duplicates} duplicate rows")
        
        print("\n👁️  Sample Records (first 3):")
        print(df.head(3).to_dict('records'))
        
        # Check for specific columns that might be interesting
        if 'code' in df.columns or 'ecsf_code' in df.columns:
            code_col = 'code' if 'code' in df.columns else 'ecsf_code'
            print(f"\n📌 Unique {code_col}: {df[code_col].nunique()}")
        
        if 'level' in df.columns:
            print("\n📊 Distribution by level:")
            print(df['level'].value_counts())
        
        if 'category' in df.columns or 'mission' in df.columns:
            cat_col = 'category' if 'category' in df.columns else 'mission'
            print(f"\n📊 Distribution by {cat_col}:")
            print(df[cat_col].value_counts())
        
    elif isinstance(data, dict):
        print(f"\n📊 Number of top-level keys: {len(data.keys())}")
        print(f"📊 Top-level keys: {list(data.keys())}")
        
        # Analyze each top-level key
        print("\n" + "=" * 80)
        print("ANALYZING EACH TOP-LEVEL KEY:")
        print("=" * 80)
        
        for key in data.keys():
            print(f"\n🔑 Key: '{key}'")
            print(f"   Type: {type(data[key]).__name__}")
            
            # If it's a list, analyze its contents
            if isinstance(data[key], list):
                print(f"   Count: {len(data[key])} items")
                
                if len(data[key]) > 0:
                    # Check structure of first item
                    first_item = data[key][0]
                    print(f"   Item type: {type(first_item).__name__}")
                    
                    if isinstance(first_item, dict):
                        print(f"   Item fields: {list(first_item.keys())}")
                        
                        # Show sample item
                        print(f"\n   📋 Sample item from '{key}':")
                        for field, value in first_item.items():
                            if isinstance(value, str) and len(value) > 100:
                                print(f"      {field}: {value[:100]}...")
                            elif isinstance(value, (list, dict)):
                                print(f"      {field}: {type(value).__name__} with {len(value)} items")
                            else:
                                print(f"      {field}: {value}")
                    else:
                        print(f"   Sample items: {data[key][:3]}")
                        
            elif isinstance(data[key], dict):
                print(f"   Sub-keys: {list(data[key].keys())}")
                print(f"   Sample: {str(data[key])[:200]}...")
            else:
                print(f"   Value: {data[key]}")
        
        # Check relationships between keys
        print("\n" + "=" * 80)
        print("RELATIONSHIPS:")
        print("=" * 80)
        
        # Look for ID or code fields that might link entities
        all_keys = list(data.keys())
        for key in all_keys:
            if isinstance(data[key], list) and len(data[key]) > 0:
                if isinstance(data[key][0], dict):
                    fields = list(data[key][0].keys())
                    # Check if any field references other top-level keys
                    for field in fields:
                        for other_key in all_keys:
                            if other_key != key and other_key in field.lower():
                                print(f"   🔗 '{key}' has field '{field}' - link to '{other_key}'")
    
    return data


def explore_job_postings(filepath):
    """Explore Job Postings data structure"""
    print("\n\n" + "=" * 80)
    print("JOB POSTINGS DATA EXPLORATION")
    print("=" * 80)
    
    data = load_json(filepath)
    
    # Check if it's a list or dict
    print(f"\n📊 Data type: {type(data).__name__}")
    
    if isinstance(data, list):
        print(f"📊 Total records: {len(data)}")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(data)
        
        print(f"\n📋 Columns ({len(df.columns)}):")
        for col in df.columns:
            print(f"  - {col}")
        
        print(f"\n📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        print("\n🔍 Column Data Types:")
        print(df.dtypes)
        
        # Check for empty string values across all columns
        print("\n⚠️  Empty String Values (null values):")
        empty_strings_found = False
        for col in df.columns:
            if df[col].dtype == 'object':  # Only check string columns
                empty_count = (df[col].str.strip() == '').sum()
                if empty_count > 0:
                    empty_strings_found = True
                    percentage = (empty_count / len(df)) * 100
                    print(f"  - {col}: {empty_count} empty strings ({percentage:.1f}%)")
        
        if not empty_strings_found:
            print("  No empty string values found")
        
        print("\n🔄 Duplicate Records:")
        duplicates = df.duplicated().sum()
        print(f"  {duplicates} duplicate rows")
        
        print("\n👁️  Sample Records (first 2):")
        for i, record in enumerate(df.head(2).to_dict('records')):
            print(f"\n  Record {i+1}:")
            for key, value in record.items():
                # Truncate long text fields
                if isinstance(value, str) and len(value) > 100:
                    print(f"    {key}: {value[:100]}...")
                else:
                    print(f"    {key}: {value}")
        
        # Analyze text fields (likely containing skills)
        text_columns = df.select_dtypes(include=['object']).columns
        print(f"\n📝 Text Columns ({len(text_columns)}):")
        for col in text_columns:
            avg_length = df[col].astype(str).str.len().mean()
            print(f"  - {col}: avg length = {avg_length:.0f} chars")
        
        # Check for location/country data
        location_cols = [col for col in df.columns if any(x in col.lower() for x in ['location', 'country', 'city'])]
        if location_cols:
            print(f"\n🌍 Location Distribution (top 10 in '{location_cols[0]}'):")
            print(df[location_cols[0]].value_counts().head(10))
        
        # Check for status/state columns
        status_cols = [col for col in df.columns if col == 'Job State']
        if status_cols:
            for col in status_cols:
                print(f"\n📌 Status Distribution ('{col}'):")
                print(df[col].value_counts())
                
                # Check for non-LISTED statuses
                non_listed = df[df[col] != 'LISTED']
                if len(non_listed) > 0:
                    print(f"\n⚠️  Found {len(non_listed)} records with status != 'LISTED':")
                    print(non_listed[col].value_counts())
                    print("\n  Sample non-LISTED records:")
                    for idx, record in non_listed.head(3).iterrows():
                        print(f"    - ID: {record.get('id', record.get('Id', 'N/A'))}, Status: {record[col]}")
                else:
                    print(f"  ✅ All records have status = 'LISTED'")
        
        # Check critical fields (Title, Description, Primary Description)
        critical_fields = ['Title', 'Description', 'Primary Description']
        existing_critical = [f for f in critical_fields if f in df.columns]
        
        if existing_critical:
            print("\n" + "=" * 80)
            print("🔑 CRITICAL FIELDS CHECK")
            print("=" * 80)
            
            for field in existing_critical:
                print(f"\n📌 Field: '{field}'")
                
                # Count nulls and empty strings
                null_count = df[field].isna().sum()
                empty_count = (df[field].str.strip() == '').sum() if df[field].dtype == 'object' else 0
                total_missing = null_count + empty_count
                valid_count = len(df) - total_missing
                
                print(f"  - Valid records: {valid_count} ({(valid_count/len(df)*100):.1f}%)")
                print(f"  - Null values: {null_count}")
                print(f"  - Empty strings: {empty_count}")
                print(f"  - Total missing: {total_missing} ({(total_missing/len(df)*100):.1f}%)")
                
                if total_missing > 0:
                    print(f"  ⚠️  Warning: {total_missing} records missing critical field '{field}'")
        
        # Language Detection
        if LANGDETECT_AVAILABLE and existing_critical:
            print("\n" + "=" * 80)
            print("🌐 LANGUAGE DETECTION")
            print("=" * 80)
            
            for field in existing_critical:
                print(f"\n📝 Detecting language in '{field}'...")
                
                # Sample detection (check first 500 non-empty records for speed)
                sample_size = min(500, len(df))
                non_empty = df[df[field].str.strip() != ''][field].head(sample_size)
                
                if len(non_empty) == 0:
                    print(f"  ⚠️  No valid text to detect language")
                    continue
                
                detected_langs = []
                for text in non_empty:
                    try:
                        lang = detect(str(text))
                        detected_langs.append(lang)
                    except LangDetectException:
                        detected_langs.append('unknown')
                
                # Count languages
                lang_counts = Counter(detected_langs)
                total_sampled = len(detected_langs)
                
                print(f"  Sampled {total_sampled} records:")
                for lang, count in lang_counts.most_common():
                    percentage = (count / total_sampled) * 100
                    emoji = "✅" if lang == 'en' else "⚠️"
                    print(f"    {emoji} {lang}: {count} ({percentage:.1f}%)")
                
                # Check if mostly English
                en_count = lang_counts.get('en', 0)
                en_percentage = (en_count / total_sampled) * 100
                
                if en_percentage < 90:
                    non_en_count = total_sampled - en_count
                    print(f"\n  ⚠️  Found {non_en_count} non-English records ({(100-en_percentage):.1f}%)")
                else:
                    print(f"\n  ✅ Mostly English ({en_percentage:.1f}%)")
        elif not LANGDETECT_AVAILABLE:
            print("\n" + "=" * 80)
            print("🌐 LANGUAGE DETECTION - SKIPPED")
            print("=" * 80)
            print("  Install langdetect: pip install langdetect")
        
    elif isinstance(data, dict):
        print(f"📊 Top-level keys: {list(data.keys())}")
        # Explore first key as sample
        first_key = list(data.keys())[0]
        print(f"\n👁️  Sample entry (key: {first_key}):")
        print(json.dumps(data[first_key], indent=2)[:500])
    
    return data


def main():
    """Main exploration function"""
    print("\n🔍 Starting Data Exploration...\n")
    
    # File paths
    ecsf_file = RAW_DATA_DIR / 'ecsf.json'
    job_postings_file = RAW_DATA_DIR / 'job_postings.json'
    
    # Check if files exist
    if not ecsf_file.exists():
        print(f"❌ ECSF file not found: {ecsf_file}")
        return
    
    if not job_postings_file.exists():
        print(f"❌ Job Postings file not found: {job_postings_file}")
        return
    
    # Explore both datasets
    try:
        ecsf_data = explore_ecsf(ecsf_file)
        job_data = explore_job_postings(job_postings_file)
        
        print("\n\n" + "=" * 80)
        print("✅ EXPLORATION COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during exploration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
