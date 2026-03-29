"""
Tests for data/fim.py — FIM gap creation and PSM formatting.
Run: python -m pytest tests/test_fim.py -v
"""
import pytest
from data.fim import (
    create_fim_sample,
    select_gap_lines,
    format_psm,
    GapError,
    FIM_BEGIN,
    FIM_HOLE,
    FIM_END,
    EOT,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CODE = "\n".join([
    "class MyClass:",           # class line — ineligible
    "    def __init__(self):",  # def line — ineligible
    "        self.x = 1",       # eligible
    "        self.y = 2",       # eligible
    "    # a comment",          # comment — ineligible
    "",                         # blank — ineligible
    "    def compute(self):",   # def line — ineligible
    "        a = self.x + 1",   # eligible
    "        b = self.y + 2",   # eligible
    "        c = a * b",         # eligible
    "        d = c - 1",         # eligible
    "        return d",          # eligible
])

# Code with 10+ eligible lines for seed-diversity test
LONG_CODE = "\n".join([
    "x = 1",
    "y = 2",
    "z = 3",
    "a = 4",
    "b = 5",
    "c = 6",
    "d = 7",
    "e = 8",
    "f = 9",
    "g = 10",
    "h = 11",
    "i = 12",
])


# ---------------------------------------------------------------------------
# Token structure tests
# ---------------------------------------------------------------------------

def test_psm_token_structure():
    """fim_text must start with FIM_BEGIN, contain FIM_HOLE and FIM_END in
    order, and end with EOT."""
    s = create_fim_sample(SAMPLE_CODE, n=1, seed=42)
    t = s["fim_text"]
    assert t.startswith(FIM_BEGIN), f"Expected FIM_BEGIN at start, got: {t[:50]!r}"
    assert FIM_HOLE in t, "FIM_HOLE not found in fim_text"
    assert FIM_END in t, "FIM_END not found in fim_text"
    assert t.endswith(EOT), f"Expected EOT at end, got: {t[-20:]!r}"

    bi = t.index(FIM_BEGIN)
    hi = t.index(FIM_HOLE)
    ei = t.index(FIM_END)
    assert bi < hi < ei, f"Wrong token order: FIM_BEGIN={bi}, FIM_HOLE={hi}, FIM_END={ei}"


def test_middle_not_in_prefix_or_suffix():
    """The masked lines must not appear in prefix or suffix."""
    s = create_fim_sample(SAMPLE_CODE, n=2, seed=42)
    middle = s["middle"]
    prefix = s["prefix"]
    suffix = s["suffix"]
    # The middle text should not be present in prefix or suffix
    assert middle not in prefix, "middle unexpectedly found in prefix"
    assert middle not in suffix, "middle unexpectedly found in suffix"


def test_prefix_plus_middle_plus_suffix_reconstructs_code():
    """prefix + middle + suffix must reconstruct the original code lines."""
    s = create_fim_sample(SAMPLE_CODE, n=2, seed=42)
    # Reassemble: prefix lines + middle lines + suffix lines
    reconstructed = s["prefix"] + "\n" + s["middle"] + "\n" + s["suffix"]
    assert reconstructed == SAMPLE_CODE, (
        f"Reconstruction mismatch.\nOriginal:      {SAMPLE_CODE!r}\n"
        f"Reconstructed: {reconstructed!r}"
    )


# ---------------------------------------------------------------------------
# Exclusion rule tests
# ---------------------------------------------------------------------------

def test_excludes_def_lines():
    """Code with only def lines as 'body' raises GapError."""
    code = "def foo():\ndef bar():\ndef baz():"
    with pytest.raises(GapError):
        create_fim_sample(code, n=1, seed=0)


def test_excludes_class_lines():
    """Code with only class lines raises GapError."""
    code = "class Foo:\nclass Bar:\nclass Baz:"
    with pytest.raises(GapError):
        create_fim_sample(code, n=1, seed=0)


def test_excludes_empty_lines():
    """Empty lines are not selected into the gap."""
    # Code: only blank lines and one eligible line
    code = "\n\nx = 1\n\n"
    # n=1 must work (picks 'x = 1'), n=2 must fail (only 1 eligible line)
    s = create_fim_sample(code, n=1, seed=0)
    assert s["middle"].strip() == "x = 1"

    with pytest.raises(GapError):
        create_fim_sample(code, n=2, seed=0)


def test_excludes_comment_lines():
    """Comment-only lines are not selected into the gap."""
    code = "# comment 1\n# comment 2\nx = 1\n# comment 3"
    s = create_fim_sample(code, n=1, seed=0)
    assert not s["middle"].strip().startswith("#"), (
        f"Middle should not be a comment line, got: {s['middle']!r}"
    )


# ---------------------------------------------------------------------------
# Determinism and diversity tests
# ---------------------------------------------------------------------------

def test_deterministic_same_seed():
    """Same seed always produces identical fim_text."""
    s1 = create_fim_sample(SAMPLE_CODE, n=1, seed=7)
    s2 = create_fim_sample(SAMPLE_CODE, n=1, seed=7)
    assert s1["fim_text"] == s2["fim_text"], "Same seed produced different output"


def test_different_seeds_may_differ():
    """Two different seeds on code with 12 eligible lines produce different start indices."""
    lines = LONG_CODE.split("\n")
    # seed=0 -> index 6, seed=1 -> index 2 (verified against random.Random.choice)
    start_a, _ = select_gap_lines(lines, n=1, seed=0)
    start_b, _ = select_gap_lines(lines, n=1, seed=1)
    assert start_a != start_b, (
        f"Expected seeds 0 and 1 to produce different start indices, both gave {start_a}"
    )


# ---------------------------------------------------------------------------
# N-line and error tests
# ---------------------------------------------------------------------------

def test_n_lines_respected():
    """create_fim_sample with n=2 produces a middle with exactly 2 lines."""
    s = create_fim_sample(SAMPLE_CODE, n=2, seed=42)
    middle_lines = s["middle"].split("\n")
    assert len(middle_lines) == 2, (
        f"Expected 2 middle lines, got {len(middle_lines)}: {middle_lines!r}"
    )
    assert s["n_lines"] == 2


def test_no_valid_gap_raises():
    """Single-line code with only a def line raises GapError."""
    code = "def foo(): pass"
    with pytest.raises(GapError):
        create_fim_sample(code, n=1, seed=0)
