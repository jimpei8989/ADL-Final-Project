from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.trainer_utils import set_seed

from datasets.dataset_dst_for_categorical_prediction import (
    DSTDatasetForDSTForCategoricalPrediction,
)
from datasets.dataset_dst_for_prediction import DSTDatasetForDSTForPrediction
from datasets.dst_collator import DSTCollator
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
        max_seq_length=512,  # TODO: extract from tokenizer
        user_token=args.user_token,
        system_token=args.system_token,
        test_mode=args.test_mode,
    )

    def to_dataloader(dataset):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=DSTCollator(tokenizer.pad_token_id),
        )

    slot_dataloader = to_dataloader(slot_dataset)

    model = DSTModel.from_pretrained(
        args.pretrained_path, model_name=args.model_name_or_path, device=args.device
    )
    model.to(args.device)
    model.eval()

    states = defaultdict(lambda: defaultdict(dict))  # (dialogue_id, turn_idx) -> slot : value
    dialogue_to_others = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(slot_dataloader, ncols=99, desc="Predicting slots"):
            batch_output = model(batch["input_ids"].to(args.device))

            for i in range(batch["input_ids"].shape[0]):
                key, encoded = batch["_key"][i], batch["_encoded"][i]
                output = batch_output.logits_by_index(i)

                if output.slot_logits > 0:
                    if schema.service_by_name[key[2]].slot_by_name[key[3]].is_categorical:
                        dialogue_to_others[key[0]].append(key[1:])
                    else:
                        begin_scores = output.begin_logits.softmax(dim=0)
                        end_scores = output.end_logits.softmax(dim=0)
                        begin_token_idx = begin_scores.argmax()
                        end_token_idx = end_scores.argmax()

                        _begin_logits = begin_scores.clone()
                        _begin_logits[end_token_idx + 1 :] = 0
                        _begin_logits[: end_token_idx - args.max_span_length + 1] = 0
                        _begin_token_idx = _begin_logits.argmax()

                        _end_logits = end_scores.clone()
                        _end_logits[begin_token_idx + args.max_span_length :] = 0
                        _end_logits[:begin_token_idx] = 0
                        _end_token_idx = _end_logits.argmax()

                        # Pair 1: begin_idx, _end_index
                        score_1 = (
                            begin_scores[begin_token_idx] * end_scores[_end_token_idx]
                            if (
                                encoded.token_to_sequence(begin_token_idx) == 0
                                and encoded.token_to_sequence(_end_token_idx) == 0
                            )
                            else 0
                        )

                        # Pair 2: _begin_idx, end_idx
                        score_2 = (
                            begin_scores[_begin_token_idx] * end_scores[end_token_idx]
                            if (
                                encoded.token_to_sequence(_begin_token_idx) == 0
                                and encoded.token_to_sequence(end_token_idx) == 0
                            )
                            else 0
                        )

                        if score_1 == 0 and score_2 == 0:
                            pass
                        else:
                            if score_1 > score_2:
                                begin_char_idx = encoded.token_to_chars(begin_token_idx).start
                                end_char_idx = encoded.token_to_chars(_end_token_idx).end
                            else:
                                begin_char_idx = encoded.token_to_chars(_begin_token_idx).start
                                end_char_idx = encoded.token_to_chars(end_token_idx).end

                            value = batch["utterance"][i][begin_char_idx : end_char_idx + 1].strip(
                                STRIP_CHARS
                            )

                            states[key[0]][key[1]][f"{key[2]}-{key[3]}"] = value

    if args.test_mode:
        print(dialogue_to_others)

    categorical_dataset = DSTDatasetForDSTForCategoricalPrediction(
        dialogue_to_others=dialogue_to_others,
        json_dir=args.test_data_dir,
        schema=schema,
        tokenizer=tokenizer,
        max_seq_length=512,  # TODO: extract from tokenizer
        user_token=args.user_token,
        system_token=args.system_token,
        test_mode=args.test_mode,
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
                categorical_outputs[key[:4]].append((answer, value_logits.item()))

    if args.test_mode:
        print(categorical_outputs)

    # Finalize categorical outputs
    for key, answers in categorical_outputs.items():
        states[key[0]][key[1]][f"{key[2]}-{key[3]}"] = max(answers, key=lambda p: p[1])[0]

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
        map(lambda s: "None" if len(s) == 0 else "|".join(f"{a}={b}" for a, b in s.items()), final)
    )
    pd.DataFrame({"id": dids, "state": final}).to_csv(args.prediction_csv, index=False)


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
