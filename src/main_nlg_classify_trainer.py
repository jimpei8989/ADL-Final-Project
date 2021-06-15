from argparse import ArgumentParser, Namespace
import os

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from utils.logger import logger
from pathlib import Path
from utils.utils import set_seed
from utils.tqdmm import tqdmm
from model.NLGClassifier import NLGClassifier
from datasets.dataset_nlg import DSTDatasetForNLG

TRAIN = "train"
DEV = "val"
SPLITS = [TRAIN, DEV]


def compute_metric(eval_pred):
    preds, labels = eval_pred

    score = ((preds > 0.5).type(torch.float) == labels).type(torch.float).mean().item()
    return {"acc": score}


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    device = torch.device(f"cuda:{args.gpu}")
    datasets = {
        split: DSTDatasetForNLG(
            path,
            mode="train",
            tokenizer=tokenizer,
        )
        for split, path in zip(SPLITS, [args.train_data, args.val_data])
    }

    model = NLGClassifier(model_name=args.backbone).to(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_args = TrainingArguments(
        str(args.ckpt_dir),
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=10,
        num_train_epochs=args.num_epoch,
        fp16=True,
        gradient_accumulation_steps=args.accumulate_steps,
        adafactor=True,
        seed=args.seed,
        load_best_model_at_end=True,
    )

    callback_list = [
        EarlyStoppingCallback(early_stopping_patience=5),
    ]
    trainer = Trainer(
        model,
        train_args,
        train_dataset=datasets[TRAIN],
        eval_dataset=datasets[DEV],
        tokenizer=tokenizer,
        compute_metrics=compute_metric,
        callbacks=callback_list,
    )

    trainer.train()


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # file PATH
    parser.add_argument(
        "--train_data", type=Path, default="./dataset/data-0610/new-train"
    )
    parser.add_argument("--val_data", type=Path, default="./dataset/data-0610/new-dev")
    parser.add_argument("--test_data", type=Path, default="./dataset/data/test_seen")
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the mathing model file.",
        default="./ckpt/default",
    )
    parser.add_argument("--opt_file", default="default.json")

    # mode
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")

    # dataloader
    parser.add_argument("--batch_size", type=int, default=8)

    # train
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--backbone", type=str, default="./models/convbert")
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--accumulate_steps", type=int, default=8)

    # Misc
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", default=24667429)
    args = parser.parse_args()

    logger.info(args)
    set_seed(args.seed)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
