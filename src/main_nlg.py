from argparse import ArgumentParser, Namespace
import json
import torch

from nlg_generate import generate
from utils.logger import logger
from pathlib import Path
from utils.utils import set_seed
from transformers import (
    AutoTokenizer,
    BlenderbotForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from datasets.dataset_nlg import DSTDatasetForNLG


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
    parser.add_argument("--predict", action="store_true")

    # model
    parser.add_argument("--pretrained", default="facebook/blenderbot-400M-distill")

    # dataloader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--history", help="Whether use history or not", action="store_true"
    )

    # Misc
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", default=24667429)
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.gpu}")

    logger.info(args)
    set_seed(args.seed)
    return args


def main(args):
    if args.train:
        pass
    if args.predict:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

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
