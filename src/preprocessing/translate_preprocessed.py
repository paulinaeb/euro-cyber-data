"""
Translate non-English Title/Description in preprocessed job postings.

Reads job_postings_preprocessed.json, detects language per record, and translates
non-English content to English. Supports sample/full mode and saves output to
data/preprocessed.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.cli_args import add_sample_mode_arguments, is_valid_sample_size
from src.utils.config import PREPROCESSED_DIR
from src.utils.sampling import sample_collection
from src.preprocessing import language_detection


def _get_translator():
    try:
        from googletrans import Translator
    except ImportError:
        return None

    return Translator()


def _detect_language(text):
    if not language_detection.LANGDETECT_AVAILABLE:
        return None

    try:
        return language_detection.detect(text)
    except language_detection.LangDetectException:
        return "unknown"


def _should_translate(lang_code):
    if lang_code in (None, "unknown"):
        return False
    return lang_code != "en"


def translate_fields(df, fields, progress_every=100):
    """Translate non-English text in specified fields. Returns df and stats."""
    translator = _get_translator()
    if translator is None:
        raise RuntimeError(
            "googletrans is not installed. Install it with: pip install googletrans==4.0.0-rc1"
        )

    stats = {field: {"translated": 0, "skipped": 0, "missing": 0, "failed": 0} for field in fields}
    cleaned_df = df.copy()

    total_records = len(cleaned_df)

    for field in fields:
        if field not in cleaned_df.columns:
            continue

        translated_values = []
        for index, value in enumerate(cleaned_df[field].tolist(), start=1):
            if progress_every and index % progress_every == 0:
                print(f"  Progress [{field}]: {index}/{total_records}")

            if pd.isna(value) or str(value).strip() == "":
                translated_values.append(value)
                stats[field]["missing"] += 1
                continue

            text = str(value)
            lang_code = _detect_language(text)
            if not _should_translate(lang_code):
                translated_values.append(text)
                stats[field]["skipped"] += 1
                continue

            try:
                translated = translator.translate(text, dest="en").text
                translated_values.append(translated)
                stats[field]["translated"] += 1
            except Exception:
                translated_values.append(text)
                stats[field]["failed"] += 1

        cleaned_df[field] = translated_values

    return cleaned_df, stats


def load_preprocessed_data(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_translated_data(data, filename):
    output_path = PREPROCESSED_DIR / filename
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}")


def print_translation_stats(stats):
    for field, field_stats in stats.items():
        print(
            f"  {field}: translated={field_stats['translated']}, "
            f"skipped={field_stats['skipped']}, missing={field_stats['missing']}, "
            f"failed={field_stats['failed']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Translate non-English Title/Description in preprocessed job postings."
    )
    add_sample_mode_arguments(
        parser,
        mode_flag="--run-mode",
        mode_dest="run_mode",
        default_mode="full",
        mode_help="Execution mode: 'sample' for quick iteration or 'full' for complete dataset (default: full).",
        sample_size_default=1000,
        sample_size_help="Number of records used when --run-mode=sample (default: 1000).",
    )
    parser.add_argument(
        "--input-file",
        default=str(PREPROCESSED_DIR / "job_postings_preprocessed.json"),
        help="Path to the preprocessed input JSON.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output filename saved under data/preprocessed.",
    )
    args = parser.parse_args()

    if not is_valid_sample_size(args.sample_size):
        print("sample-size must be greater than 0")
        return

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    if not language_detection.LANGDETECT_AVAILABLE:
        print("langdetect not installed. Install it with: pip install langdetect")
        return

    data = load_preprocessed_data(input_path)
    data = sample_collection(data, mode=args.run_mode, sample_size=args.sample_size)
    df = pd.DataFrame(data)

    fields_to_translate = ["Title", "Description"]
    translated_df, stats = translate_fields(df, fields_to_translate)

    print("Translation stats:")
    print_translation_stats(stats)

    if args.output_file:
        output_name = args.output_file
    elif args.run_mode == "sample":
        output_name = f"job_postings_preprocessed_sample_{len(df)}.json"
    else:
        output_name = "job_postings_preprocessed.json"

    save_translated_data(translated_df.to_dict("records"), output_name)


if __name__ == "__main__":
    main()
