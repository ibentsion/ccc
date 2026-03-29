"""Tests for data.dataset module — TDD RED phase."""
import pytest


def test_load_jsonl_fields():
    """First 5 records from JSONL have instruction, input, output keys."""
    from data.dataset import load_jsonl
    records = load_jsonl("python-codes-25k.jsonl")
    assert len(records) >= 5
    for record in records[:5]:
        assert "instruction" in record
        assert "input" in record
        assert "output" in record


def test_load_jsonl_file_not_found():
    """FileNotFoundError raised for non-existent path."""
    from data.dataset import load_jsonl
    with pytest.raises(FileNotFoundError):
        load_jsonl("non_existent_file_xyz.jsonl")


def test_split_no_data_loss():
    """100 synthetic records split 0.8/0.1/0.1 — total == 100."""
    from data.dataset import split_dataset
    records = [{"id": i} for i in range(100)]
    train, val, test = split_dataset(records, train=0.8, val=0.1, test=0.1, seed=42)
    assert len(train) + len(val) + len(test) == 100


def test_split_deterministic():
    """Same seed produces identical first elements."""
    from data.dataset import split_dataset
    records = [{"id": i} for i in range(100)]
    result1 = split_dataset(records, train=0.8, val=0.1, test=0.1, seed=42)
    result2 = split_dataset(records, train=0.8, val=0.1, test=0.1, seed=42)
    assert result1[0][0] == result2[0][0]


def test_split_different_seeds():
    """Different seeds produce different orderings."""
    from data.dataset import split_dataset
    records = [{"id": i} for i in range(100)]
    train_42, _, _ = split_dataset(records, train=0.8, val=0.1, test=0.1, seed=42)
    train_99, _, _ = split_dataset(records, train=0.8, val=0.1, test=0.1, seed=99)
    assert train_42[0] != train_99[0]


def test_split_bad_ratios():
    """Ratios not summing to 1.0 raise ValueError."""
    from data.dataset import split_dataset
    with pytest.raises(ValueError):
        split_dataset([], 0.5, 0.3, 0.1)


def test_split_original_not_mutated():
    """Original list is not mutated after split."""
    from data.dataset import split_dataset
    records = [{"id": i} for i in range(100)]
    original_order = [r["id"] for r in records]
    split_dataset(records, train=0.8, val=0.1, test=0.1, seed=42)
    assert [r["id"] for r in records] == original_order
