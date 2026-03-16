"""Shared sampling helpers for pipeline modules."""


def sample_collection(data, mode='full', sample_size=1000):
    """
    Return either full data or a deterministic head sample.

    Supported inputs:
    - Python sequences (list/tuple/str): uses slicing
    - pandas objects (Series/DataFrame): uses .head(...)
    """
    if mode not in {'sample', 'full'}:
        raise ValueError("mode must be 'sample' or 'full'")

    if sample_size <= 0:
        raise ValueError('sample_size must be greater than 0')

    if mode == 'full':
        return data

    limit = min(sample_size, len(data))

    if hasattr(data, 'head'):
        return data.head(limit)

    return data[:limit]
