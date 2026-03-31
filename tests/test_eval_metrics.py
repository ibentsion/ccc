import sys
import types
from unittest.mock import MagicMock

import pytest

from training.eval_metrics import exact_match, edit_similarity, run_eval
from data.fim import FIM_END, EOT


# --- exact_match ---

def test_exact_match_identical():
    assert exact_match("foo\nbar", "foo\nbar") == 1.0


def test_exact_match_different():
    assert exact_match("foo\nbar", "foo\nbaz") == 0.0


def test_exact_match_empty():
    assert exact_match("", "") == 1.0


# --- edit_similarity ---

def test_edit_similarity_identical():
    assert edit_similarity("abc", "abc") == 1.0


def test_edit_similarity_completely_different():
    assert edit_similarity("abc", "xyz") == 0.0


def test_edit_similarity_partial():
    result = edit_similarity("kitten", "sitting")
    assert 0 < result < 1


def test_edit_similarity_empty():
    assert edit_similarity("", "") == 1.0


# --- run_eval ---

def _make_run_eval_fixtures():
    import torch

    # Mock tokenizer: decode returns a FIM string with known middle text
    middle_text = "x = 1"
    tokenizer = MagicMock()
    tokenizer.decode.return_value = f"prefix{FIM_END}{middle_text}{EOT}suffix"

    # Mock model: model.model.generate returns dummy token ids
    model = MagicMock()
    model.model.generate.return_value = torch.tensor([[1, 2, 3]])

    # Fake dataloader yielding one batch
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "labels": torch.tensor([[1, 2]]),
    }
    dataloader = [batch]

    ground_truths = [middle_text]
    return model, tokenizer, dataloader, ground_truths


def test_run_eval_returns_expected_keys():
    torch = pytest.importorskip("torch")
    model, tokenizer, dataloader, ground_truths = _make_run_eval_fixtures()
    result = run_eval(model, tokenizer, dataloader, ground_truths, max_new_tokens=32)
    assert "exact_match" in result
    assert "edit_sim" in result


def test_run_eval_metrics_are_floats():
    torch = pytest.importorskip("torch")
    model, tokenizer, dataloader, ground_truths = _make_run_eval_fixtures()
    result = run_eval(model, tokenizer, dataloader, ground_truths, max_new_tokens=32)
    assert isinstance(result["exact_match"], float)
    assert isinstance(result["edit_sim"], float)


def test_run_eval_calls_generate_with_greedy_params():
    torch = pytest.importorskip("torch")
    model, tokenizer, dataloader, ground_truths = _make_run_eval_fixtures()
    run_eval(model, tokenizer, dataloader, ground_truths, max_new_tokens=64)
    call_kwargs = model.model.generate.call_args[1]
    assert call_kwargs.get("do_sample") is False
    assert call_kwargs.get("num_beams") == 1
    assert call_kwargs.get("max_new_tokens") == 64
