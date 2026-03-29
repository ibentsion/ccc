"""
FIM gap creation and PSM formatting for DeepSeek-Coder fine-tuning.

Exports:
    FIM_BEGIN, FIM_HOLE, FIM_END, EOT  — DeepSeek-Coder FIM token constants
    GapError                            — raised when no valid gap exists
    select_gap_lines(lines, n, seed)    — returns (start_idx, end_idx) of gap
    format_psm(lines, start, end)       — returns PSM dict
    create_fim_sample(code, n, seed)    — top-level convenience function
"""

import random
import re

# ---------------------------------------------------------------------------
# FIM token constants
# Copy these exact bytes — do NOT retype; the special chars are Unicode.
# ---------------------------------------------------------------------------
FIM_BEGIN = "<｜fim▁begin｜>"
FIM_HOLE  = "<｜fim▁hole｜>"
FIM_END   = "<｜fim▁end｜>"
EOT       = "<|EOT|>"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GapError(ValueError):
    """Raised when no valid N-consecutive-eligible-line gap can be found."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_ineligible(line: str) -> bool:
    """Return True if a line must not be included in a FIM gap."""
    stripped = line.strip()
    if stripped == "":
        return True
    if stripped.startswith("#"):
        return True
    if re.match(r"^\s*(def |class )", line):
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_gap_lines(lines, n, seed):
    """
    Select N consecutive eligible lines to use as the FIM gap.

    Parameters
    ----------
    lines : list[str]
        Lines of the code file (split on "\\n").
    n : int
        Number of consecutive eligible lines required.
    seed : int
        Random seed for deterministic but varied selection.

    Returns
    -------
    (start_idx, end_idx) : tuple[int, int]
        start_idx is inclusive, end_idx is exclusive (end_idx == start_idx + n).

    Raises
    ------
    GapError
        If no run of n consecutive eligible lines exists.
    """
    eligible = [i for i, l in enumerate(lines) if not _is_ineligible(l)]

    # Find all starting positions that begin a run of n consecutive integers
    # in the eligible list.
    valid_starts = []
    j = 0
    while j < len(eligible):
        # Try to extend a run from eligible[j]
        run_start = j
        while j + 1 < len(eligible) and eligible[j + 1] == eligible[j] + 1:
            j += 1
        # Run covers indices run_start..j (inclusive) in 'eligible'
        run_length = j - run_start + 1
        if run_length >= n:
            # All windows of size n within this run are valid starts
            for k in range(run_start, run_start + run_length - n + 1):
                valid_starts.append(eligible[k])
        j += 1

    if not valid_starts:
        raise GapError(
            f"No valid gap of {n} consecutive eligible lines found in "
            f"{len(lines)}-line code"
        )

    start = random.Random(seed).choice(valid_starts)
    return (start, start + n)


def format_psm(lines, start, end):
    """
    Format a PSM (Prefix-Suffix-Middle) FIM training sample.

    Parameters
    ----------
    lines : list[str]
        All lines of the code.
    start : int
        Inclusive start of the gap (from select_gap_lines).
    end : int
        Exclusive end of the gap.

    Returns
    -------
    dict with keys:
        "fim_text" — full PSM string
        "prefix"   — lines before the gap, joined with "\\n"
        "suffix"   — lines after the gap, joined with "\\n"
        "middle"   — the masked lines, joined with "\\n"
    """
    prefix = "\n".join(lines[:start])
    suffix = "\n".join(lines[end:])
    middle = "\n".join(lines[start:end])
    fim_text = f"{FIM_BEGIN}{prefix}{FIM_HOLE}{suffix}{FIM_END}{middle}{EOT}"
    return {
        "fim_text": fim_text,
        "prefix": prefix,
        "suffix": suffix,
        "middle": middle,
    }


def create_fim_sample(code, n, seed=42):
    """
    Create a PSM FIM training sample from a code string.

    Parameters
    ----------
    code : str
        Complete Python source code as a single string.
    n : int
        Number of consecutive lines to mask as the gap.
    seed : int, optional
        Random seed (default 42).

    Returns
    -------
    dict with keys from format_psm plus "n_lines": n.

    Raises
    ------
    GapError
        If no valid gap of n consecutive eligible lines exists.
    """
    lines = code.split("\n")
    start, end = select_gap_lines(lines, n, seed)
    result = format_psm(lines, start, end)
    result["n_lines"] = n
    return result
