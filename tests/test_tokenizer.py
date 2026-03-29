import pytest

transformers = pytest.importorskip("transformers")
from transformers import AutoTokenizer

from data.fim import FIM_BEGIN, FIM_HOLE, FIM_END, EOT
from training.tokenizer import load_tokenizer, TokenizedCollator

MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"


def _make_sample(prefix: str, suffix: str, middle: str) -> dict:
    fim_text = FIM_BEGIN + prefix + FIM_HOLE + suffix + FIM_END + middle + EOT
    return {
        "fim_text": fim_text,
        "prefix": prefix,
        "suffix": suffix,
        "middle": middle,
        "n_lines": 1,
    }


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(MODEL_NAME)


def test_load_tokenizer_config(tokenizer):
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.padding_side == "left"


def test_collator_output_keys(tokenizer):
    collator = TokenizedCollator(tokenizer, max_seq_length=128)
    sample = _make_sample("x = 1\n", "\nprint(x)", "x += 1")
    out = collator([sample])
    assert set(out.keys()) == {"input_ids", "attention_mask", "labels"}


def test_prompt_masking_single(tokenizer):
    collator = TokenizedCollator(tokenizer, max_seq_length=256)
    sample = _make_sample("x = 1\n", "\nprint(x)", "x += 1")
    out = collator([sample])

    labels = out["labels"][0]
    input_ids = out["input_ids"][0]

    # Determine the expected middle token range
    prefix_part = FIM_BEGIN + sample["prefix"] + FIM_HOLE + sample["suffix"] + FIM_END
    prefix_ids = tokenizer.encode(prefix_part, add_special_tokens=False)
    middle_ids = tokenizer.encode(sample["middle"], add_special_tokens=False)

    pad_len = (input_ids == tokenizer.pad_token_id).sum().item()
    middle_start = pad_len + len(prefix_ids)
    middle_end = middle_start + len(middle_ids)

    # Everything before middle (padding + prompt) must be -100
    assert (labels[:middle_start] == -100).all(), "Expected -100 for all pre-middle tokens"
    # Middle tokens must have real label values (not -100)
    assert (labels[middle_start:middle_end] != -100).all(), "Middle tokens must not be masked"


def test_prompt_masking_batch(tokenizer):
    collator = TokenizedCollator(tokenizer, max_seq_length=256)
    short_sample = _make_sample("a = 1\n", "\npass", "a += 1")
    long_sample = _make_sample(
        "x = 1\ny = 2\nz = 3\n",
        "\nprint(x)\nprint(y)\nprint(z)\nreturn x + y + z",
        "x += 1",
    )
    out = collator([short_sample, long_sample])

    input_ids = out["input_ids"]
    labels = out["labels"]

    # Count -100 labels per sample
    masked_short = (labels[0] == -100).sum().item()
    masked_long = (labels[1] == -100).sum().item()

    # Both padded to same length; shorter sample gets more left-padding → more -100 labels
    pad_short = (input_ids[0] == tokenizer.pad_token_id).sum().item()
    pad_long = (input_ids[1] == tokenizer.pad_token_id).sum().item()
    assert pad_short >= pad_long, "Shorter sample should have at least as many padding tokens"
    assert masked_short >= masked_long, "Shorter sample should have at least as many -100 labels"
