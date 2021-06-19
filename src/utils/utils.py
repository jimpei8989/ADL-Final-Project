import random
import numpy as np
import torch

from rouge_score import rouge_scorer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class metrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        rec = {"rouge1": 0, "rougeL": 0}
        for pred, gt in zip(predictions, labels):
            pred = self.tokenizer.decode(pred, skip_special_tokens=True)
            gt = self.tokenizer.decode(
                [0 if v == -100 else v for v in gt], skip_special_tokens=True
            )
            if pred == "":
                pred = "empty"

            score = self.scorer.score(gt, pred)
            rec = {k: rec[k] + v.fmeasure for k, v in score.items()}

        return {k: v / len(predictions) for k, v in rec.items()}
