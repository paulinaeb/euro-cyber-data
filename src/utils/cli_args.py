"""Shared CLI argument helpers for script entry points."""


def add_sample_mode_arguments(
    parser,
    mode_flag='--run-mode',
    mode_dest='run_mode',
    default_mode='full',
    mode_help=None,
    sample_size_default=1000,
    sample_size_help=None,
):
    """Add sample/full mode arguments to an argparse parser."""
    if mode_help is None:
        mode_help = (
            "Execution mode: 'sample' for quick iteration or 'full' "
            "for complete dataset."
        )

    if sample_size_help is None:
        sample_size_help = (
            f"Number of records used when {mode_flag}=sample "
            f"(default: {sample_size_default})."
        )

    parser.add_argument(
        mode_flag,
        dest=mode_dest,
        choices=['sample', 'full'],
        default=default_mode,
        help=mode_help,
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=sample_size_default,
        help=sample_size_help,
    )

    return parser


def is_valid_sample_size(sample_size):
    """Return True when sample size is valid for sample/full mode scripts."""
    return sample_size > 0
