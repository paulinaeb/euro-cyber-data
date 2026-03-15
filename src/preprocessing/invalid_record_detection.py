"""Helpers for detecting invalid job-posting records."""

from src.preprocessing.language_detection import invalid_content_mask


DEFAULT_CRITICAL_FIELDS = ('Title', 'Description', 'Primary Description', 'Skill')


def get_existing_critical_fields(df, critical_fields=None):
    """Return the subset of critical fields that exist in the DataFrame."""
    fields = critical_fields or DEFAULT_CRITICAL_FIELDS
    return [field for field in fields if field in df.columns]


def get_all_critical_fields_invalid_mask(df, critical_fields=None):
    """Return a mask for rows where all existing critical fields are invalid."""
    existing_fields = get_existing_critical_fields(df, critical_fields)
    if not existing_fields:
        return None, []

    mask = None
    for field in existing_fields:
        field_mask = invalid_content_mask(df[field])
        mask = field_mask if mask is None else (mask & field_mask)

    return mask, existing_fields


def find_all_critical_fields_invalid_records(df, critical_fields=None):
    """Return invalid records where all existing critical fields are invalid."""
    mask, existing_fields = get_all_critical_fields_invalid_mask(df, critical_fields)
    if mask is None:
        return df.iloc[0:0].copy(), existing_fields
    return df[mask].copy(), existing_fields
