from argparse import ArgumentParser, Namespace
from pathlib import Path
import json

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
from utils.utils import set_seed, get_dataset_kwargs, add_tokens


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(f"predictions type = {type(predictions)}, shape = {predictions.shape}")
    print(f"labels type = {type(labels)}, shape = {labels.shape}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--train_data_dir", type=Path, default="dataset/data-0625/train/")
    parser.add_argument("--eval_data_dir", type=Path, default="dataset/data-0625/dev/")
    parser.add_argument("--schema_json", type=Path, default="dataset/data/schema.json")
    parser.add_argument("--ckpt_dir", type=Path, default="./ckpt/DST/default/")

    # optimizer
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    # dataset
    parser.add_argument("--user_token", help="use this after ensuring token is in vocab.txt")
    parser.add_argument("--system_token", help="use this after ensuring token is in vocab.txt")
    parser.add_argument("--strategy", choices=["turn", "segment"], default="segment")
    parser.add_argument("--last_user_turn_only", action="store_true")
    parser.add_argument("--reserved_for_latter", type=int, default=48)
    parser.add_argument("--overlap_turns", type=int, default=4)
    parser.add_argument("--no_ensure_user_on_both_ends", action="store_true")
    parser.add_argument("--dataset_type", choices=["slot", "categorical", "span"], nargs="+")

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", default=24296674, type=int)

    # training
    parser.add_argument("--model_name_or_path", default="models/convbert-dg")
    parser.add_argument("--pool", action="store_true", default=False)
    parser.add_argument("--accumulate_steps", type=int, default=16)
    parser.add_argument("--no_adafactor", action="store_true", default=False)
    parser.add_argument("--no_fp16", action="store_true", default=False)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--early_stopping", type=int, default=3)

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


def save_args(args) -> None:
    class PathEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Path):
                return str(obj)
            return json.JSONEncoder.default(self, obj)

    args.ckpt_dir.mkdir(parents=True)
    json.dump(
        vars(args),
        (args.ckpt_dir / "arguments.json").open("w"),
        indent=4,
        cls=PathEncoder,
    )


def main(args):
    logger.info(args)
    save_args(args)
    set_seed(args.seed)

    model = DSTModel(model_name=args.model_name_or_path, pool=args.pool)
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

    tokenizer = add_tokens(tokenizer, args.user_token, args.system_token)
    dataset_kwargs = get_dataset_kwargs(args, max_length=model.max_position_embeddings)

    trainer = DSTTrainer(
        train_data_dir=args.train_data_dir,
        eval_data_dir=args.eval_data_dir,
        schema_json=args.schema_json,
        tokenizer=tokenizer,
        train_dataloader_cls=dataloader_cls.to_train_dataloader,
        eval_dataloader_cls=dataloader_cls.to_eval_dataloader,
        dataset_kwargs=dataset_kwargs,
        model=model,
        args=train_args,
        compute_metrics=compute_metrics,
        dataset_type=args.dataset_type,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.early_stopping))

    trainer.train()


if __name__ == "__main__":
    main(parse_args())
