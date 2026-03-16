"""
Data Preprocessing Script
Cleans, transforms, and prepares raw data for the pipeline.
Saves preprocessed data to data/preprocessed/

"""

import json
import html
import re
import pandas as pd
from pathlib import Path
import argparse
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import RAW_DATA_DIR, PREPROCESSED_DIR
from src.utils.cli_args import add_sample_mode_arguments, is_valid_sample_size
from src.utils.markup_detection import (
    COLLAPSE_NEWLINES_PATTERN,
    DECORATIVE_SEPARATOR_PATTERN,
    find_records_with_markup,
    GENDER_MARKER_PATTERN,
    get_detected_markup_types,
    HTML_TAG_PATTERN,
    INLINE_CODE_PATTERN,
    MARKDOWN_BOLD_PATTERN,
    MARKDOWN_LINK_PATTERN,
    MARKDOWN_UNDERSCORE_PATTERN,
    NEWLINE_SPACING_PATTERN,
    NEWLINE_TO_PERIOD_PATTERN,
    remove_gender_marker_tokens,
    QUOTE_CHARACTER_PATTERN,
    RAW_URL_PATTERN,
    remove_emoji_like_unicode,
    WHITESPACE_PATTERN,
)
from src.utils.sampling import sample_collection
from src.preprocessing.invalid_record_detection import (
    get_all_critical_fields_invalid_mask,
)
from src.preprocessing.language_detection import detect_language_distribution


JOB_POSTINGS_COLUMNS_TO_DROP = [
    'Insight',
    'Job State',
    'Detail URL',
    'Company Name',
    'Company Description',
    'Company Website',
    'Company Logo',
    'Company Apply Url',
    'Employee Count',
    'Headquarters',
    'Company Founded',
    'Specialties',
    'Hiring Manager Title',
    'Hiring Manager Subtitle',
    'Hiring Manager Title Insight',
    'Hiring Manager Profile',
    'Hiring Manager Image',
    'Poster Id',
]

SKILL_PREFIX_PATTERN = re.compile(r'^\s*skills:\s*', flags=re.IGNORECASE)
SKILL_SUFFIX_MORE_PATTERN = re.compile(r'\s*,\s*\+\s*\d+\s+more\s*$', flags=re.IGNORECASE)
SKILL_PROFILE_MATCH_PATTERN = re.compile(
    r'^\s*\d+\s+of\s+\d+\s+skills\s+match\s+your\s+profile\s*-\s*you\s+may\s+be\s+a\s+good\s+fit\s*$',
    flags=re.IGNORECASE,
)

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


def drop_unneeded_job_posting_columns(df):
    """Remove columns that are not needed for this analysis.
    The function only drops columns that actually exist in the input DataFrame. """
    existing_columns_to_drop = [
        column for column in JOB_POSTINGS_COLUMNS_TO_DROP if column in df.columns
    ]

    if not existing_columns_to_drop:
        return df.copy(), []

    cleaned_df = df.drop(columns=existing_columns_to_drop).copy()
    return cleaned_df, existing_columns_to_drop


def print_sample_record(df, label, max_fields=8, max_text_length=120):
    """Print a compact sample record to show how the data looks at a given step."""
    print(f"\n  {label}")

    if df.empty:
        print("    No records available")
        return

    record = df.iloc[0].to_dict()
    preview = {}
    for index, (key, value) in enumerate(record.items()):
        if index >= max_fields:
            break

        if isinstance(value, str) and len(value) > max_text_length:
            preview[key] = f"{value[:max_text_length]}..."
        else:
            preview[key] = value

    print(f"    {preview}")


def clean_markup_from_text(value):
    """Remove URLs/markup and quote noise while preserving apostrophes inside words."""
    if pd.isna(value):
        return value

    text = html.unescape(str(value))
    text = MARKDOWN_LINK_PATTERN.sub(r'\1', text)
    text = RAW_URL_PATTERN.sub(' ', text)
    text = HTML_TAG_PATTERN.sub(' ', text)
    text = MARKDOWN_BOLD_PATTERN.sub(r'\1', text)
    text = MARKDOWN_UNDERSCORE_PATTERN.sub(r'\1', text)
    text = INLINE_CODE_PATTERN.sub(r'\1', text)
    text = remove_gender_marker_tokens(text)
    text = QUOTE_CHARACTER_PATTERN.sub('', text)
    text = DECORATIVE_SEPARATOR_PATTERN.sub('', text)
    text = remove_emoji_like_unicode(text)
    text = WHITESPACE_PATTERN.sub(' ', text)
    text = NEWLINE_SPACING_PATTERN.sub('\n', text)
    text = COLLAPSE_NEWLINES_PATTERN.sub('\n', text)
    text = NEWLINE_TO_PERIOD_PATTERN.sub('. ', text)
    return text.strip()


def clean_gender_markers_in_columns(df, columns=('Title', 'Description')):
    """Remove gender marker variants from selected text columns and return per-column counts."""
    cleaned_df = df.copy()
    cleaned_counts = {}

    for column in columns:
        if column not in cleaned_df.columns:
            continue

        texts = cleaned_df[column].fillna('').astype(str)
        marker_mask = texts.str.contains(GENDER_MARKER_PATTERN, regex=True)
        cleaned_counts[column] = int(marker_mask.sum())

        if cleaned_counts[column] == 0:
            continue

        cleaned_df.loc[marker_mask, column] = texts[marker_mask].apply(
            lambda text: WHITESPACE_PATTERN.sub(' ', remove_gender_marker_tokens(text)).strip()
        )

    return cleaned_df, cleaned_counts


def _clean_skill_value(value):
    """Normalize Skill text by removing boilerplate wrappers and profile-match sentences."""
    if pd.isna(value):
        return value

    text = str(value).strip()

    if SKILL_PROFILE_MATCH_PATTERN.fullmatch(text):
        return ''

    text = SKILL_PREFIX_PATTERN.sub('', text)
    text = SKILL_SUFFIX_MORE_PATTERN.sub('', text)
    return WHITESPACE_PATTERN.sub(' ', text).strip()


def clean_skill_feature(df, column='Skill'):
    """Clean the Skill column and return per-rule counts."""
    cleaned_df = df.copy()

    if column not in cleaned_df.columns:
        return cleaned_df, {
            'prefix_removed': 0,
            'suffix_removed': 0,
            'profile_match_cleared': 0,
        }

    texts = cleaned_df[column].fillna('').astype(str)
    stats = {
        'prefix_removed': int(texts.str.contains(SKILL_PREFIX_PATTERN, regex=True).sum()),
        'suffix_removed': int(texts.str.contains(SKILL_SUFFIX_MORE_PATTERN, regex=True).sum()),
        'profile_match_cleared': int(texts.str.contains(SKILL_PROFILE_MATCH_PATTERN, regex=True).sum()),
    }

    cleaned_df[column] = cleaned_df[column].apply(_clean_skill_value)
    return cleaned_df, stats


def save_markup_cleaning_examples(before_records, after_df, detection_result, before_path, after_path, column='Description', sample_count=20):
    """Save before/after examples for records whose descriptions contained markup-like content."""
    sample_before = before_records.head(sample_count)
    before_export = []
    after_export = []

    for index, record in sample_before.iterrows():
        detected_types = get_detected_markup_types(index, detection_result)
        before_export.append({
            'Title': record.get('Title'),
            'Location': record.get('Location'),
            'Primary Description': record.get('Primary Description'),
            'Description': record.get(column),
            'Detected Markup Types': detected_types,
        })

        after_record = after_df.loc[index]
        after_export.append({
            'Title': after_record.get('Title'),
            'Location': after_record.get('Location'),
            'Primary Description': after_record.get('Primary Description'),
            'Description': after_record.get(column),
            'Detected Markup Types Before Cleaning': detected_types,
        })

    with open(before_path, 'w', encoding='utf-8') as before_file:
        json.dump(before_export, before_file, ensure_ascii=False, indent=2)

    with open(after_path, 'w', encoding='utf-8') as after_file:
        json.dump(after_export, after_file, ensure_ascii=False, indent=2)


def clean_description_markup(df, column='Description', before_examples_path=None, after_examples_path=None):
    """
    Detect and remove markup-like content from the description column.

    This step cleans URLs, HTML tags/entities, and simple markdown-like patterns
    while preserving the remaining readable text.
    """
    if column not in df.columns:
        return df.copy(), df.iloc[0:0].copy(), 0

    markup_records, detection_result = find_records_with_markup(df, column=column)
    cleaned_df = df.copy()
    cleaned_df[column] = cleaned_df[column].apply(clean_markup_from_text)

    remaining_markup_records, _ = find_records_with_markup(cleaned_df, column=column)

    if before_examples_path is not None and after_examples_path is not None:
        save_markup_cleaning_examples(
            markup_records,
            cleaned_df,
            detection_result,
            before_examples_path,
            after_examples_path,
            column=column,
        )

    return cleaned_df, markup_records, len(remaining_markup_records)


def preprocess_ecsf(data):
    """
    Preprocess ECSF data
    
    TODO: Implement based on exploration findings
    - Standardize field names
    - Extract relevant fields
    """
    print("🔧 Preprocessing ECSF data...")
    
    pass


def preprocess_job_postings(data, markup_examples_before_path=None, markup_examples_after_path=None):
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
    print_sample_record(df, 'Sample record at start:')

    # Step 1: drop columns that are not needed for the current pipeline stage.
    # Doing this first simplifies the record shape for all following preprocessing steps.
    cleaned_df, dropped_columns = drop_unneeded_job_posting_columns(df)

    if dropped_columns:
        print(f"  Dropped columns: {dropped_columns}")
        print(f"  Remaining columns: {len(cleaned_df.columns)}")
    else:
        print("  No configured columns found to drop")

    # Step 2: detect and save "before" examples from original state (before any cleaning).
    markup_records, detection_result = find_records_with_markup(cleaned_df, column='Description')
    
    if markup_examples_before_path is not None and len(markup_records) > 0:
        # Save the original, unmodified records as "before" examples for validation
        sample_before = markup_records.head(20)
        before_export = []
        
        for index, record in sample_before.iterrows():
            detected_types = get_detected_markup_types(index, detection_result)
            before_export.append({
                'Title': record.get('Title'),
                'Location': record.get('Location'),
                'Primary Description': record.get('Primary Description'),
                'Description': record.get('Description'),
                'Detected Markup Types': detected_types,
            })
        
        with open(markup_examples_before_path, 'w', encoding='utf-8') as f:
            json.dump(before_export, f, ensure_ascii=False, indent=2)
        print(f"  Saved description markup examples before cleaning: {markup_examples_before_path}")

    # Step 3: remove gender marker variants from title/description text.
    cleaned_df, gender_marker_counts = clean_gender_markers_in_columns(cleaned_df)
    if gender_marker_counts:
        for column, count in gender_marker_counts.items():
            print(f"  {column} records with gender markers removed: {count}")
    else:
        print("  No target columns found for gender marker cleaning")

    # Step 4: remove markup-like content from descriptions and save "after" examples.
    cleaned_df, markup_records_cleaned, remaining_markup_count = clean_description_markup(
        cleaned_df,
        before_examples_path=None,  # Already saved in Step 2
        after_examples_path=markup_examples_after_path,
    )
    print(f"  Description records with markup details: detected={len(markup_records)}, remaining after cleaning={remaining_markup_count}")
    if markup_examples_after_path is not None:
        print(f"  Saved description markup examples after cleaning: {markup_examples_after_path}")

    # Step 5: clean Skill feature boilerplate.
    cleaned_df, skill_clean_stats = clean_skill_feature(cleaned_df, column='Skill')
    if 'Skill' in cleaned_df.columns:
        print(
            "  Skill cleanup: "
            f"prefix_removed={skill_clean_stats['prefix_removed']}, "
            f"suffix_removed={skill_clean_stats['suffix_removed']}, "
            f"profile_match_cleared={skill_clean_stats['profile_match_cleared']}"
        )
    else:
        print("  Skill column not found for Skill cleanup")

    # Step 6: remove rows where every critical field is invalid.
    cleaned_df, invalid_records, checked_fields = remove_records_with_all_critical_fields_invalid(cleaned_df)

    if checked_fields:
        print(f"  Critical fields checked: {checked_fields}")
        print(f"  Invalid records removed: {len(invalid_records)}")
        print(f"  Remaining records: {len(cleaned_df)}")
    else:
        print("  No critical fields found for invalid-record detection")

    print_sample_record(cleaned_df, 'Sample record at end:')

    return cleaned_df.to_dict('records')


def save_preprocessed_data(data, filename):
    """Save preprocessed data to JSON"""
    output_path = PREPROCESSED_DIR / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved: {output_path}")


def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description='Run job postings preprocessing in sample or full mode.')
    add_sample_mode_arguments(
        parser,
        mode_flag='--run-mode',
        mode_dest='run_mode',
        default_mode='full',
        mode_help="Execution mode: 'sample' for quick iteration or 'full' for complete dataset (default: full).",
        sample_size_default=1000,
        sample_size_help='Number of records used when --run-mode=sample (default: 1000).',
    )
    args = parser.parse_args()

    if not is_valid_sample_size(args.sample_size):
        print('❌ sample-size must be greater than 0')
        return

    print("⚙️  Starting Data Preprocessing...\n")
    print(f"  Run mode: {args.run_mode}")
    if args.run_mode == 'sample':
        print(f"  Sample size: {args.sample_size}")
    
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

    original_job_count = len(job_data)
    job_data = sample_collection(job_data, mode=args.run_mode, sample_size=args.sample_size)
    print(f"  Job postings loaded: {len(job_data)} / {original_job_count}")
    
    # ecsf_preprocessed = preprocess_ecsf(ecsf_data)
    if args.run_mode == 'sample':
        markup_examples_before_path = PREPROCESSED_DIR / f'job_postings_description_markup_before_sample_{len(job_data)}.json'
        markup_examples_after_path = PREPROCESSED_DIR / f'job_postings_description_markup_after_sample_{len(job_data)}.json'
    else:
        markup_examples_before_path = PREPROCESSED_DIR / 'job_postings_description_markup_before_full.json'
        markup_examples_after_path = PREPROCESSED_DIR / 'job_postings_description_markup_after_full.json'

    job_preprocessed = preprocess_job_postings(
        job_data,
        markup_examples_before_path=markup_examples_before_path,
        markup_examples_after_path=markup_examples_after_path,
    )

    # Save preprocessed data
    # save_preprocessed_data(ecsf_preprocessed, 'ecsf_preprocessed.json')
    if args.run_mode == 'sample':
        output_name = f"job_postings_preprocessed_sample_{len(job_data)}.json"
    else:
        output_name = 'job_postings_preprocessed.json'

    save_preprocessed_data(job_preprocessed, output_name)


if __name__ == '__main__':
    main()
