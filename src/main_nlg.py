from argparse import ArgumentParser, Namespace
import os
import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from transformers import (
    AutoTokenizer,
    BlenderbotForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from datasets.dataset_nlg import DSTDatasetForNLG
from datasets.dataset_nlg_end import DSTDatasetForNLGEnd
from utils.utils import metrics, set_seed
from datasets.dataset_nlg_end import DSTDatasetForNLGEnd
from nlg_generate import generate, generate_oneside
from utils.logger import logger


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # file PATH
    parser.add_argument("--train_data", type=Path, default="./dataset/data/train")
    parser.add_argument("--val_data", type=Path, default="./dataset/data/dev")
    parser.add_argument("--test_data", type=Path, default="./dataset/data/test_seen")
    parser.add_argument("--schema", type=Path, default="./dataset/data/schema.json")
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the mathing model file.",
        default="./ckpt/default",
    )
    parser.add_argument("--opt_file", default="default.json")

    # mode
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_begin", action="store_true")
    parser.add_argument("--train_end", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--predict_begin", action="store_true")
    parser.add_argument("--predict_end", action="store_true")

    # model
    parser.add_argument("--pretrained", default="facebook/blenderbot-400M-distill")

    # dataloader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--history", help="Whether use history or not", action="store_true"
    )

    # trainer
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--num_epoch", type=int, default=100)

    # Misc
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", default=24667429)
    args = parser.parse_args()
    args.device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logger.info(args)
    set_seed(args.seed)
    return args


def main(args):
    if args.train:
        pass

    if args.train_end ^ args.train_begin:
        which_side = "end" if args.train_end else "beginning"
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
        if "blenderbot" in args.pretrained:
            model = BlenderbotForConditionalGeneration.from_pretrained(args.pretrained)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained)
        logger.info(model.config)

        train_data = DSTDatasetForNLGEnd(
            args.train_data,
            tokenizer=tokenizer,
            mode="train",
            get_full_history=False,
            which_side=which_side,
        )
        val_data = DSTDatasetForNLGEnd(
            args.val_data,
            tokenizer=tokenizer,
            mode="train",
            get_full_history=False,
            which_side=which_side,
        )
        train_args = Seq2SeqTrainingArguments(
            args.ckpt_dir,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            adafactor=True,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            weight_decay=args.wd,
            num_train_epochs=args.num_epoch,
            predict_with_generate=True,
            fp16=True,
            gradient_accumulation_steps=8,
            load_best_model_at_end=True,
            save_total_limit=5,
        )
        trainer = Seq2SeqTrainer(
            model,
            train_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            tokenizer=tokenizer,
            compute_metrics=metrics(tokenizer),
            callbacks=[EarlyStoppingCallback(3)],
        )
        trainer.train()

    if args.predict_end ^ args.predict_begin:
        which_side = "end" if args.predict_end else "beginning"
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.ckpt_dir, local_files_only=True
        )
        test_data = DSTDatasetForNLGEnd(
            args.test_data,
            tokenizer=tokenizer,
            mode="test",
            get_full_history=False,
            which_side=which_side,
        )
        test_loader = DataLoader(
            test_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=test_data.collate_fn,
        )
        result = generate_oneside(
            model, test_loader, tokenizer, device=args.device, which_side=which_side
        )
        json.dump(result, open(args.opt_file, "w"))

    if args.predict:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

        if "blenderbot" in args.pretrained:
            model = BlenderbotForConditionalGeneration.from_pretrained(args.pretrained)
        elif "gpt" in args.pretrained:
            model = AutoModelForCausalLM.from_pretrained(args.pretrained)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained)
        logger.info(model.config)

        dataset = DSTDatasetForNLG(
            args.test_data,
            tokenizer=tokenizer,
            mode="test",
            get_full_history=args.history,
        )

        result = generate(
            model, dataset, batch_size=args.batch_size, device=args.device
        )
        json.dump(result, open(args.opt_file, "w"))


if __name__ == "__main__":
    main(parse_args())
