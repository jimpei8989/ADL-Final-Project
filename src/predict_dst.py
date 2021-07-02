import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets.dataset_dst_for_categorical_prediction import (
    DSTDatasetForDSTForCategoricalPrediction,
)
from datasets.dataset_dst_for_prediction import DSTDatasetForDSTForPrediction
from datasets.dst_collator import DSTCollator
from datasets.schema import Schema
from DST.DSTModel import DSTModel
from utils.io import json_load
from utils.utils import set_seed, get_dataset_kwargs, add_tokens

STRIP_CHARS = ",.:;!?\"'@#$%^&*()| \t\n"


def load_args(args_path: Path) -> Namespace:
    if args_path is not None:
        train_args = Namespace(**json_load(args_path))
        return train_args
    return None


def main(args):
    # Add this line to avoid tokenizer raising warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_seed(args.seed)
    train_args = load_args(args.train_args_path)

    schema = Schema.load_json(args.schema_json)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = DSTModel.from_pretrained(
        args.pretrained_dir / "pytorch_model.bin",
        model_name=args.model_name_or_path,
        pool=getattr(train_args, "pool", False),
        device=args.device,
    )
    model.to(args.device)
    model.eval()

    dataset_kwargs = get_dataset_kwargs(
        train_args if train_args is not None else args, max_length=model.max_position_embeddings
    )
    tokenizer = add_tokens(
        tokenizer,
        train_args.user_token if train_args is not None else args.user_token,
        train_args.system_token if train_args is not None else args.system_token,
    )

    slot_dataset = DSTDatasetForDSTForPrediction(
        json_dir=args.test_data_dir,
        schema=schema,
        tokenizer=tokenizer,
        test_mode=args.test_mode,
        **dataset_kwargs,
    )

    def to_dataloader(dataset):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=DSTCollator(tokenizer.pad_token_id),
        )

    slot_dataloader = to_dataloader(slot_dataset)

    states = defaultdict(lambda: defaultdict(dict))  # (dialogue_id, turn_idx) -> slot : value
    dialogue_to_others = defaultdict(list)

    span_lengthes = Counter()

    with torch.no_grad():
        for batch in tqdm(slot_dataloader, ncols=99, desc="Predicting slots"):
            batch_output = model(batch["input_ids"].to(args.device))

            for i in range(batch["input_ids"].shape[0]):
                key, encoded = batch["_key"][i], batch["_encoded"][i]
                output = batch_output.logits_by_index(i)

                if output.slot_logits > 0:
                    if schema.service_by_name[key[3]].slot_by_name[key[4]].is_categorical:
                        dialogue_to_others[key[0]].append(key[1:])
                    else:
                        begin_scores = output.begin_logits.softmax(dim=0)
                        end_scores = output.end_logits.softmax(dim=0)

                        begin_token_idx, end_token_idx = max(
                            (
                                (i, j)
                                for i in range(begin_scores.shape[0])
                                if encoded.token_to_sequence(i) == 0
                                for j in range(
                                    i, min(i + args.max_span_length, begin_scores.shape[0])
                                )
                                if encoded.token_to_sequence(j) == 0
                            ),
                            key=lambda p: begin_scores[p[0]] * end_scores[p[1]],
                        )
                        span_lengthes.update([end_token_idx - begin_token_idx + 1])

                        begin_char_idx = encoded.token_to_chars(begin_token_idx).start
                        end_char_idx = encoded.token_to_chars(end_token_idx).end
                        states[key[0]][key[2]][f"{key[3]}-{key[4]}"] = batch["utterance"][i][
                            begin_char_idx : end_char_idx + 1
                        ].strip(STRIP_CHARS)

    if args.test_mode:
        print(dialogue_to_others)

    categorical_dataset = DSTDatasetForDSTForCategoricalPrediction(
        dialogue_to_others=dialogue_to_others,
        json_dir=args.test_data_dir,
        schema=schema,
        tokenizer=tokenizer,
        test_mode=args.test_mode,
        **dataset_kwargs,
    )
    categorical_dataloader = to_dataloader(categorical_dataset)

    # (did, turn_idx, service, slot) -> List[_nswer, logit]
    categorical_outputs = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(categorical_dataloader, ncols=99, desc="Predicting categorical"):
            batch_output = model(batch["input_ids"].to(args.device))

            for i in range(batch["input_ids"].shape[0]):
                key, answer = batch["_key"][i], batch["answer"][i]
                value_logits = batch_output.logits_by_index(i).value_logits
                categorical_outputs[key[:5]].append((answer, value_logits.item()))

    if args.test_mode:
        print(categorical_outputs)

    # Finalize categorical outputs
    for key, answers in categorical_outputs.items():
        states[key[0]][key[2]][f"{key[3]}-{key[4]}"] = max(answers, key=lambda p: p[1])[0]

    # Finalize states
    final_states = defaultdict(dict)

    for did in states:
        for i in sorted(states[did]):
            final_states[did].update(states[did][i])

    if args.test_mode:
        print(final_states)

    # Dump to output file
    dids, final = zip(*final_states.items())
    final = list(
        map(
            lambda s: "None"
            if len(s) == 0
            else "|".join(
                f"{a.lower()}={b.replace(',', '_').lower()}"
                for a, b in sorted(s.items(), key=lambda x: x[0])
            ),
            final,
        )
    )
    pd.DataFrame({"id": dids, "state": final}).to_csv(args.prediction_csv, index=False)


def parse_args():
    parser = ArgumentParser()

    # Model
    parser.add_argument("--model_name_or_path", default="bert-base-uncased")
    parser.add_argument("--pretrained_dir", required=True, type=Path)
    parser.add_argument("--train_args_path", type=Path)

    # Dataset
    parser.add_argument("--test_data_dir", type=Path, default="dataset/data/test_seen")
    parser.add_argument("--schema_json", type=Path, default="dataset/data/schema.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--user_token", help="use this after ensuring token is in vocab.txt")
    parser.add_argument("--system_token", help="use this after ensuring token is in vocab.txt")
    parser.add_argument("--strategy", choices=["turn", "segment"], default="segment")
    parser.add_argument("--last_user_turn_only", action="store_true")
    parser.add_argument("--reserved_for_latter", type=int, default=48)
    parser.add_argument("--overlap_turns", type=int, default=4)
    parser.add_argument("--no_ensure_user_on_both_ends", action="store_true")

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
