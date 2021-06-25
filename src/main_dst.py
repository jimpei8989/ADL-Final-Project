from argparse import ArgumentParser, Namespace
from pathlib import Path
import torch
import numpy as np
import random

from torch.utils.data import DataLoader

from transformers import (
    PreTrainedTokenizerBase,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from utils.logger import logger
from datasets.dst_collator import DSTCollator
from DST.DSTTrainer import DSTTrainer
from DST.DSTModel import DSTModel


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(f"predictions type = {type(predictions)}, shape = {predictions.shape}")
    print(f"labels type = {type(labels)}, shape = {labels.shape}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--train_data_dir", type=Path, default="dataset/data-0614/train/")
    parser.add_argument("--eval_data_dir", type=Path, default="dataset/data-0614/dev/")
    parser.add_argument("--schema_json", type=Path, default="dataset/data/schema.json")
    parser.add_argument("--ckpt_dir", type=Path, default="./ckpt/DST/default/")

    # optimizer
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    # data loader
    parser.add_argument("--user_token", help="use this after ensuring token is in vocab.txt")
    parser.add_argument("--system_token", help="use this after ensuring token is in vocab.txt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", default=24296674, type=int)

    # training
    parser.add_argument("--model_name_or_path", default="models/convbert-dg")
    parser.add_argument("--accumulate_steps", type=int, default=16)
    parser.add_argument("--no_adafactor", action="store_true", default=False)
    parser.add_argument("--no_fp16", action="store_true", default=False)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--early_stopping", type=int, default=5)

    args = parser.parse_args()
    args.seed %= 2 ** 32

    return args


class DataloaderCLS:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, batch_size: int, num_workers: int):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers

    def to_train_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=DSTCollator(pad_value=self.tokenizer.pad_token_id),
        )

    def to_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=DSTCollator(pad_value=self.tokenizer.pad_token_id),
        )


def special_token_check(token: str, tokenizer: PreTrainedTokenizerBase):
    if token is not None:
        ids = tokenizer.convert_tokens_to_ids([token])
        if len(ids) == 1 and ids[0] != tokenizer.unk_token_id:
            return True
        return False
    return True


def main(args):
    logger.info(args)
    set_seed(args.seed)

    model = DSTModel(model_name=args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_args = TrainingArguments(
        args.ckpt_dir,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        num_train_epochs=args.num_epoch,
        gradient_accumulation_steps=args.accumulate_steps,
        adafactor=not args.no_adafactor,
        label_names=["slot_labels", "value_labels", "begin_labels", "end_labels"],
        load_best_model_at_end=True,
        fp16=not args.no_fp16,
        max_steps=2 ** 20,
    )

    dataloader_cls = DataloaderCLS(tokenizer, args.batch_size, args.num_workers)

    assert special_token_check(args.user_token, tokenizer)
    assert special_token_check(args.system_token, tokenizer)
    dataset_kwargs = {}
    dataset_kwargs["user_token"] = args.user_token
    dataset_kwargs["system_token"] = args.system_token
    if args.user_token is not None:
        tokenizer.add_special_tokens({"additional_special_tokens": [args.user_token]})
    if args.system_token is not None:
        tokenizer.add_special_tokens({"additional_special_tokens": [args.system_token]})

    trainer = DSTTrainer(
        train_data_dir=args.train_data_dir,
        eval_data_dir=args.eval_data_dir,
        schema_json=args.schema_json,
        tokenizer=tokenizer,
        train_dataloader_cls=dataloader_cls.to_train_dataloader,
        eval_dataloader_cls=dataloader_cls.to_eval_dataloader,
        dataset_kwargs=dataset_kwargs,
        for_slot_kwargs={"max_seq_length": model.max_position_embeddings},
        for_categorical_kwargs={"max_seq_length": model.max_position_embeddings},
        for_span_kwargs={"max_seq_length": model.max_position_embeddings},
        model=model,
        args=train_args,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.early_stopping))

    trainer.train()


if __name__ == "__main__":
    main(parse_args())
