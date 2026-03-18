"""Shared helpers for detecting markup-like content in text fields."""

import re

import pandas as pd


HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
HTML_ENTITY_PATTERN = re.compile(r'&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);')
MARKDOWN_LINK_PATTERN = re.compile(r'\[([^\]]+)\]\([^)]+\)')
MARKDOWN_BOLD_PATTERN = re.compile(r'\*\*([^*]+)\*\*')
MARKDOWN_UNDERSCORE_PATTERN = re.compile(r'__([^_]+)__')
INLINE_CODE_PATTERN = re.compile(r'`([^`]+)`')
# Accept standard URLs and scraped variants with malformed protocol separators,
RAW_URL_PATTERN = re.compile(r'(?i)(?:https?\s*[:\-]\s*/\s*/\s*|www\.)[^\s<>()\[\]{}"\']+')
# Broken/partial links that survive cleaning
BROKEN_OR_PARTIAL_URL_PATTERN = re.compile(
    r'(?i)(?:https?://\s*)?(?:www\.\s*)?\.[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/[^\s)\]}">\']*)?(?:\?[^\s)\]}">\']*)?'
)
EMAIL_PATTERN = re.compile(r'(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b')
# Keep this conservative: 9+ digits avoids removing short numeric fragments.
PHONE_NUMBER_PATTERN = re.compile(r'(?<!\w)\+?\d[\d\s().\-]{7,}\d(?!\w)')
# Leftover star clusters such as `**`, `*****`, or markdown remnants attached to words.
ASTERISK_CLUSTER_PATTERN = re.compile(r'\*{2,}')
# Redacted placeholders often found where contact details were pre-masked at source.
REDACTED_PLACEHOLDER_PATTERN = re.compile(r'(?<!\w)\*{3,}(?!\w)')
EMPTY_WRAPPER_PATTERN = re.compile(r'\(\s*\)|\[\s*\]|\{\s*\}')
# Matches markers like m/f, (m / f), m/f/x, m-w-d, f/h (any case, optional spaces/parentheses).
GENDER_MARKER_PATTERN = re.compile(
    r'(?i)(?<!\w)[\(\[]?\s*[mfwxdh]\s*(?:[\/|\\-]?\s*[mfwxdh]\s*){1,2}[\)\]]?(?!\w)'
)
# Remove escaped/double quotes and standalone apostrophes, but keep apostrophes inside words.
QUOTE_CHARACTER_PATTERN = re.compile(r'\\"|"|(?<!\w)\'|\'(?!\w)')
WHITESPACE_PATTERN = re.compile(r'[ \t]+')
NEWLINE_SPACING_PATTERN = re.compile(r' *\n *')
EXTRA_BLANK_LINES_PATTERN = re.compile(r'\n{3,}')
COLLAPSE_NEWLINES_PATTERN = re.compile(r'\n+')
NEWLINE_TO_PERIOD_PATTERN = re.compile(r'(?<![.!?,:;])\n')
# Matches lines consisting entirely of repeated decorator characters (e.g. ----, ***, ###, ~~~, /////, ______)
# Includes Unicode dash separators like U+2014, U+2015, and U+2E3B.
DECORATIVE_SEPARATOR_PATTERN = re.compile(
    r'(?m)^[ \t]*(?:-{3,}|\*{3,}|#{3,}|~{3,}|/{3,}|_{3,}|[\u2014\u2015\u2e3b]{1,})[ \t]*$'
)


MARKUP_PATTERNS = {
    'HTML tags': HTML_TAG_PATTERN,
    'HTML entities': HTML_ENTITY_PATTERN,
    # Use detection-specific patterns without capture groups to avoid pandas warnings.
    'Markdown links': re.compile(r'\[[^\]]+\]\([^)]+\)'),
    'Markdown formatting': re.compile(r'(?:\*\*[^*]+\*\*|__[^_]+__|`[^`]+`)'),
    'URLs': RAW_URL_PATTERN,
    'Broken or partial URLs': BROKEN_OR_PARTIAL_URL_PATTERN,
    'Email addresses': EMAIL_PATTERN,
    'Phone numbers': PHONE_NUMBER_PATTERN,
    'Asterisk clusters': ASTERISK_CLUSTER_PATTERN,
    'Redacted placeholders': REDACTED_PLACEHOLDER_PATTERN,
    'Gender markers': GENDER_MARKER_PATTERN,
    'Quote characters': QUOTE_CHARACTER_PATTERN,
    'Decorative separators': DECORATIVE_SEPARATOR_PATTERN,
}

# Emoji and similar pictographic Unicode ranges (including joiners/selectors).
EMOJI_LIKE_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # Flags
    "\U0001F300-\U0001F5FF"  # Symbols & pictographs
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F680-\U0001F6FF"  # Transport & map symbols
    "\U0001F700-\U0001F77F"  # Alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric shapes extended
    "\U0001F800-\U0001F8FF"  # Supplemental arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental symbols and pictographs
    "\U0001FA00-\U0001FAFF"  # Symbols and pictographs extended-A
    "\U00002700-\U000027BF"  # Dingbats
    "\U00002600-\U000026FF"  # Misc symbols
    "\U0000FE0F"              # Variation selector-16
    "\U0000200D"              # Zero width joiner
    "\U000020E3"              # Combining enclosing keycap
    "]+",
    flags=re.UNICODE,
)


def get_markup_detection_result(series):
    """Return per-pattern masks plus a combined mask for markup-like content."""
    texts = series.fillna('').astype(str)
    pattern_masks = {
        label: texts.str.contains(pattern, regex=True)
        for label, pattern in MARKUP_PATTERNS.items()
    }

    any_markup_mask = pd.Series(False, index=texts.index)
    for mask in pattern_masks.values():
        any_markup_mask = any_markup_mask | mask

    return {
        'texts': texts,
        'pattern_masks': pattern_masks,
        'any_markup_mask': any_markup_mask,
    }


def find_records_with_markup(df, column='Description'):
    """Return records whose target column contains markup-like patterns."""
    if column not in df.columns:
        return df.iloc[0:0].copy(), None

    detection_result = get_markup_detection_result(df[column])
    markup_records = df[detection_result['any_markup_mask']].copy()
    return markup_records, detection_result


def get_markup_counts(detection_result):
    """Summarize markup counts for each detection pattern."""
    if detection_result is None:
        return {}

    return {
        label: int(mask.sum())
        for label, mask in detection_result['pattern_masks'].items()
    }


def get_detected_markup_types(row_index, detection_result):
    """Return the markup pattern labels detected for a specific row index."""
    if detection_result is None:
        return []

    detected_types = []
    for label, mask in detection_result['pattern_masks'].items():
        if bool(mask.loc[row_index]):
            detected_types.append(label)

    return detected_types


def remove_emoji_like_unicode(text):
    """Remove emoji and similar pictographic Unicode characters from text."""
    return EMOJI_LIKE_PATTERN.sub(' ', text)


def remove_gender_marker_tokens(text):
    """Remove gender marker tokens such as m/f, m/f/x, (m / f), and related variants."""
    return GENDER_MARKER_PATTERN.sub(' ', text)


def remove_email_addresses(text):
    """Remove email addresses from text."""
    return EMAIL_PATTERN.sub(' ', text)


def remove_phone_numbers(text):
    """Remove likely phone numbers from text."""
    return PHONE_NUMBER_PATTERN.sub(' ', text)


def remove_broken_or_partial_urls(text):
    """Remove malformed or partially masked links that still look like URLs."""
    return BROKEN_OR_PARTIAL_URL_PATTERN.sub(' ', text)


def remove_asterisk_clusters(text):
    """Remove leftover asterisk clusters such as ** or ****** from text."""
    return ASTERISK_CLUSTER_PATTERN.sub(' ', text)


def remove_redacted_placeholders(text):
    """Remove masked placeholder tokens like ****** from text."""
    return REDACTED_PLACEHOLDER_PATTERN.sub(' ', text)


def remove_empty_wrappers(text):
    """Remove empty parentheses/brackets/braces left after other cleanup steps."""
    return EMPTY_WRAPPER_PATTERN.sub(' ', text)
