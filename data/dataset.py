"""JSONL loading and deterministic train/val/test splitting for CodeComplete."""
import json
import random
from typing import Dict, List, Tuple


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file and return a list of dicts.

    Each non-blank line is parsed as JSON. Blank lines are silently skipped.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of dicts, one per non-blank line.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
            return records
    except IOError:
        raise FileNotFoundError("JSONL file not found: {}".format(path))


def split_dataset(
    records: List[Dict],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Split records into train, val, and test sets deterministically.

    The split is reproducible: the same seed always yields the same partition.
    The original list is never mutated.

    Args:
        records: List of dicts to split.
        train: Fraction for training set (default 0.8).
        val: Fraction for validation set (default 0.1).
        test: Fraction for test set (default 0.1).
        seed: Random seed for shuffling (default 42).

    Returns:
        Tuple of (train_list, val_list, test_list).

    Raises:
        ValueError: If train + val + test does not sum to 1.0 (tolerance 1e-6).
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train)
    n_val = int(n * val)

    train_list = shuffled[:n_train]
    val_list = shuffled[n_train:n_train + n_val]
    test_list = shuffled[n_train + n_val:]

    return train_list, val_list, test_list
