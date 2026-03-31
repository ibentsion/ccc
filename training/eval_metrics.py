import difflib

import torch

from data.fim import FIM_END, EOT


def exact_match(predicted: str, expected: str) -> float:
    return 1.0 if predicted.strip() == expected.strip() else 0.0


def edit_similarity(predicted: str, expected: str) -> float:
    if predicted == "" and expected == "":
        return 1.0
    return difflib.SequenceMatcher(None, predicted, expected).ratio()


def run_eval(model, tokenizer, dataloader, ground_truths: list[str], max_new_tokens: int) -> dict:
    em_scores, es_scores = [], []
    gt_iter = iter(ground_truths)

    model.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            generated = model.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
            )
            for ids in generated:
                decoded = tokenizer.decode(ids, skip_special_tokens=False)
                if FIM_END in decoded:
                    after_fim = decoded.split(FIM_END, 1)[1]
                    middle = after_fim.split(EOT, 1)[0]
                else:
                    middle = ""
                gt = next(gt_iter)
                em_scores.append(exact_match(middle, gt))
                es_scores.append(edit_similarity(middle, gt))

    return {
        "exact_match": float(sum(em_scores) / len(em_scores)) if em_scores else 0.0,
        "edit_sim": float(sum(es_scores) / len(es_scores)) if es_scores else 0.0,
    }
