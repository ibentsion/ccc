import torch
from transformers import AutoTokenizer

from data.fim import FIM_BEGIN, FIM_HOLE, FIM_END


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


class TokenizedCollator:
    def __init__(self, tokenizer, max_seq_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch: list[dict]) -> dict:
        fim_texts = [sample["fim_text"] for sample in batch]
        encoded = self.tokenizer(
            fim_texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        labels = input_ids.clone()

        for i, sample in enumerate(batch):
            prefix_part = FIM_BEGIN + sample["prefix"] + FIM_HOLE + sample["suffix"] + FIM_END
            prefix_ids = self.tokenizer.encode(prefix_part, add_special_tokens=False)
            pad_len = (input_ids[i] == self.tokenizer.pad_token_id).sum().item()
            labels[i, : pad_len + len(prefix_ids)] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }
