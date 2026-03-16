"""
Data Exploration
Analyzes raw JSON files (ECSF and Job Postings) to understand structure,
quality, and characteristics before preprocessing.
"""

import json
import pandas as pd
from pathlib import Path
import argparse
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import RAW_DATA_DIR
from src.utils.cli_args import add_sample_mode_arguments, is_valid_sample_size
from src.utils.sampling import sample_collection
from src.preprocessing.invalid_record_detection import (
    find_all_critical_fields_invalid_records,
)
from src.preprocessing.language_detection import (
    LANGDETECT_AVAILABLE,
    detect_language_distribution,
    invalid_content_mask,
)

if not LANGDETECT_AVAILABLE:
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


def explore_job_postings(
    filepath,
    run_mode='full',
    run_sample_size=1000,
    language_mode='sample',
    language_sample_size=1000,
):
    """Explore Job Postings data structure"""
    print("\n\n" + "=" * 80)
    print("JOB POSTINGS DATA EXPLORATION")
    print("=" * 80)
    
    data = load_json(filepath)
    
    # Check if it's a list or dict
    print(f"\n📊 Data type: {type(data).__name__}")
    
    if isinstance(data, list):
        original_total = len(data)
        data = sample_collection(data, mode=run_mode, sample_size=run_sample_size)

        if run_mode == 'sample':
            print(f"📊 Records loaded for analysis: {len(data)} / {original_total}")
        else:
            print(f"📊 Total records: {len(data)}")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(data)
        
        print(f"\n📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        print("\n🔍 Column Data Types:")
        print(df.dtypes)
        
        # Check for empty string values across all columns
        print("\n⚠️  Missing Values (invalid content):")
        empty_strings_found = False
        for col in df.columns:
            if df[col].dtype == 'object':  # Only check string columns
                empty_count = invalid_content_mask(df[col]).sum()
                if empty_count > 0:
                    empty_strings_found = True
                    percentage = (empty_count / len(df)) * 100
                    print(f"  - {col}: {empty_count} ({percentage:.3f}%)")
        
        if not empty_strings_found:
            print("  No missing values (invalid content) found")
        
        print("\n🔄 Duplicate Records:")
        duplicates_exact = df.duplicated().sum()
        print(f"  - Exact duplicates (all columns): {duplicates_exact}")

        comparison_columns = [col for col in df.columns if col != 'Scraped At']
        if comparison_columns:
            duplicates_ignore_scraped = df.duplicated(subset=comparison_columns).sum()
            print(f"  - Duplicates ignoring 'Scraped At': {duplicates_ignore_scraped}")
        
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
        
        # Check critical fields (including Skill when present)
        critical_fields = ['Title', 'Description', 'Primary Description', 'Skill']
        existing_critical = [f for f in critical_fields if f in df.columns]
        
        if existing_critical:
            print("\n" + "=" * 80)
            print("🔑 CRITICAL FIELDS CHECK")
            print("=" * 80)
            
            for field in existing_critical:
                print(f"\n📌 Field: '{field}'")
                
                # Count nulls and empty/invalid strings
                null_count = df[field].isna().sum()
                empty_count = invalid_content_mask(df[field]).sum() if df[field].dtype == 'object' else 0
                total_missing = null_count + empty_count
                valid_count = len(df) - total_missing
                
                print(f"  - Valid records: {valid_count} ({(valid_count/len(df)*100):.1f}%)")
                print(f"  - Null values: {null_count}")
                print(f"  - Missing values (invalid content): {empty_count}")
                print(f"  - Total missing: {total_missing} ({(total_missing/len(df)*100):.1f}%)")
                
                if total_missing > 0:
                    print(f"  ⚠️  Warning: {total_missing} records missing critical field '{field}'")
        
        # Records where ALL critical fields are empty
        all_empty, all_critical_cols = find_all_critical_fields_invalid_records(df)
        if all_critical_cols:
            print("\n" + "=" * 80)
            print("🚨 RECORDS WITH ALL CRITICAL FIELDS EMPTY")
            print("=" * 80)

            print(f"\n  Fields checked: {all_critical_cols}")
            print(f"  Records with ALL fields empty: {len(all_empty)} ({(len(all_empty)/len(df)*100):.2f}%)")
            
            if len(all_empty) > 0:
                print("\n  Sample records:")
                for i, (_, record) in enumerate(all_empty.head(3).iterrows()):
                    print(f"    Record {i+1}: {dict(list(record.items())[:5])}")
            else:
                print("  ✅ No records have all critical fields empty")
        
        # Records where Skill AND Description are both invalid
        skill_desc_cols = [f for f in ['Skill', 'Description'] if f in df.columns]
        if len(skill_desc_cols) == 2:
            print("\n" + "=" * 80)
            print("⚠️  RECORDS WITH SKILL AND DESCRIPTION BOTH INVALID")
            print("=" * 80)

            skill_invalid = invalid_content_mask(df['Skill'])
            desc_invalid = invalid_content_mask(df['Description'])

            print(f"\n  Records with BOTH Skill and Description invalid: {(skill_invalid & desc_invalid).sum()} ({((skill_invalid & desc_invalid).sum()/len(df)*100):.2f}%)")
            print(f"  Records with only Skill invalid:                  {(skill_invalid & ~desc_invalid).sum()} ({((skill_invalid & ~desc_invalid).sum()/len(df)*100):.2f}%)")
            print(f"  Records with only Description invalid:            {(~skill_invalid & desc_invalid).sum()} ({((~skill_invalid & desc_invalid).sum()/len(df)*100):.2f}%)")
            print(f"  Records with NEITHER invalid (both present):      {(~skill_invalid & ~desc_invalid).sum()} ({((~skill_invalid & ~desc_invalid).sum()/len(df)*100):.2f}%)")

            both_invalid_df = df[skill_invalid & desc_invalid]
            if len(both_invalid_df) > 0:
                print("\n  Sample records (both invalid):")
                for i, (_, record) in enumerate(both_invalid_df.head(3).iterrows()):
                    print(f"    Record {i+1}: {dict(list(record.items())[:5])}")

        # Language Detection
        if language_sample_size == 0:
            print("\n" + "=" * 80)
            print("🌐 LANGUAGE DETECTION - SKIPPED")
            print("=" * 80)
            print("  Skipped because --language-sample-size is set to 0")
        elif LANGDETECT_AVAILABLE:
            print("\n" + "=" * 80)
            print("🌐 LANGUAGE DETECTION")
            print("=" * 80)
            print(f"  Mode: {language_mode}")
            if language_mode == 'sample':
                print(f"  Sample size per field: {language_sample_size}")
            
            # Fields to check for language
            fields_to_check = ['Title', 'Description', 'Primary Description', 'Skill']
            existing_fields = [f for f in fields_to_check if f in df.columns]
            
            for field in existing_fields:
                print(f"\n📝 Detecting language in '{field}'...")

                result = detect_language_distribution(
                    df[field],
                    mode=language_mode,
                    sample_size=language_sample_size,
                )

                total_sampled = result['sampled_records']
                if total_sampled == 0:
                    print(f"  ⚠️  No valid text to detect language")
                    continue

                lang_counts = result['language_counts']
                
                print(f"  Sampled {total_sampled} records:")
                for lang, count in lang_counts.most_common():
                    percentage = (count / total_sampled) * 100
                    emoji = "✅" if lang == 'en' else "⚠️"
                    print(f"    {emoji} {lang}: {count} ({percentage:.1f}%)")
                
                # Check if mostly English
                en_percentage = result['english_percentage']
                en_count = lang_counts.get('en', 0)
                
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
    parser = argparse.ArgumentParser(description='Explore ECSF and job posting raw data.')

    # Dataset-level sampling (shared behavior with preprocess_data.py)
    add_sample_mode_arguments(
        parser,
        mode_flag='--run-mode',
        mode_dest='run_mode',
        default_mode='full',
        mode_help="Execution mode for dataset analysis: 'sample' or 'full' (default: full).",
        sample_size_default=1000,
        sample_size_help='Number of job-posting records used when --run-mode=sample (default: 1000).',
    )

    # Language-detection sampling can be configured separately.
    parser.add_argument(
        '--language-mode',
        choices=['sample', 'full'],
        default='sample',
        help="Language detection mode: 'sample' (default) or 'full'.",
    )
    parser.add_argument(
        '--language-sample-size',
        type=int,
        default=None,
        help='Sample size per field for language detection. Use 0 to skip language detection. Defaults to --sample-size if not provided.',
    )
    args = parser.parse_args()

    if not is_valid_sample_size(args.sample_size):
        print('❌ sample-size must be greater than 0')
        return

    language_sample_size = args.language_sample_size
    if language_sample_size is None:
        language_sample_size = args.sample_size

    if language_sample_size < 0:
        print('❌ language-sample-size must be 0 or greater')
        return

    print("\n🔍 Starting Data Exploration...\n")
    print(f"  Run mode: {args.run_mode}")
    if args.run_mode == 'sample':
        print(f"  Dataset sample size: {args.sample_size}")
    if language_sample_size == 0:
        print("  Language detection: disabled (--language-sample-size 0)")
    else:
        print(f"  Language mode: {args.language_mode}")
    if args.language_mode == 'sample' and language_sample_size > 0:
        print(f"  Language sample size per field: {language_sample_size}")
    
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
        job_data = explore_job_postings(
            job_postings_file,
            run_mode=args.run_mode,
            run_sample_size=args.sample_size,
            language_mode=args.language_mode,
            language_sample_size=language_sample_size,
        )
        
        print("\n\n" + "=" * 80)
        print("✅ EXPLORATION COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during exploration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
