from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import torch
import torch.nn.functional as nnf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase
from transformers.trainer_utils import set_seed

from datasets.dataset_dst_for_prediction import DSTDatasetForDSTForPrediction
from datasets.schema import Schema
from DST.DSTModel import DSTModel

STRIP_CHARS = ",.:;!?\"'@#$%^&*()| \t\n"


def special_token_check(token: str, tokenizer: PreTrainedTokenizerBase):
    if token is not None:
        ids = tokenizer.convert_tokens_to_ids([token])
        if len(ids) == 1 and ids[0] != tokenizer.unk_token_id:
            return True
        return False
    return True


def main(args):
    set_seed(args.seed)

    schema = Schema.load_json(args.schema_json)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    assert special_token_check(args.user_token, tokenizer)
    assert special_token_check(args.system_token, tokenizer)

    dataset_kwargs = {}
    dataset_kwargs["user_token"] = args.user_token
    dataset_kwargs["system_token"] = args.system_token
    if args.user_token is not None:
        tokenizer.add_special_tokens({"additional_special_tokens": [args.user_token]})
    if args.system_token is not None:
        tokenizer.add_special_tokens({"additional_special_tokens": [args.system_token]})

    slot_dataset = DSTDatasetForDSTForPrediction(
        json_dir=args.test_data_dir,
        schema=schema,
        tokenizer=tokenizer,
        max_seq_length=512 - 10,  # TODO: extract from tokenizer
        user_token=args.user_token,
        system_token=args.system_token,
        test_mode=args.test_mode,
    )
    slot_dataloader = DataLoader(
        slot_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = DSTModel.from_pretrained(
        args.pretrained_path, model_name=args.model_name_or_path, device=args.device
    )
    model.to(args.device)
    model.eval()

    slot_outputs = {}  # (dialogue_id, turn_idx) -> model_output)
    with torch.no_grad():
        for batch in tqdm(slot_dataloader, ncols=99, desc="Predicting slots"):
            batch_output = model(batch["input_ids"].to(args.device))

            for i in range(batch["input_ids"].shape[0]):
                slot_outputs[
                    (batch["dialogue_id"][i], batch["service"][i], batch["slot"][i])
                ] = batch_output.logits_by_index(i)


def parse_args():
    parser = ArgumentParser()

    # Model
    parser.add_argument("--model_name_or_path", default="bert-base-uncased")
    parser.add_argument("--pretrained_path", required=True, type=Path)

    # Dataset
    parser.add_argument("--test_data_dir", type=Path, default="dataset/data/test_seen")
    parser.add_argument("--schema_json", type=Path, default="dataset/data/schema.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--user_token", help="use this after ensuring token is in vocab.txt")
    parser.add_argument("--system_token", help="use this after ensuring token is in vocab.txt")

    # Prediction
    parser.add_argument("--prediction_csv", type=Path, default="prediction.csv")
    parser.add_argument("--max_span_length", type=int, default=32)

    # Misc
    parser.add_argument("--no_gpu", dest="gpu", action="store_false")
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--seed", default=24296674, type=int)
    parser.add_argument("--test_mode", action="store_true")

    args = parser.parse_args()

    args.device = (
        torch.device("cpu")
        if not args.gpu
        else torch.device("cuda")
        if args.gpu_id is None
        else torch.device(f"cuda:{args.gpu_id}")
    )

    return args


if __name__ == "__main__":
    main(parse_args())
