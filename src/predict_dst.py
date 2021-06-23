from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BatchEncoding
from transformers.trainer_utils import set_seed

from datasets.dataset_dst_for_prediction import DSTDatasetForDSTForPrediction
from datasets.schema import Schema
from DST.DSTModel import DSTModel


def main(args):
    set_seed(args.seed)

    schema = Schema.load_json(args.schema_json)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    dataset = DSTDatasetForDSTForPrediction(json_dir=args.test_data_dir, test_mode=args.test_mode)

    model = DSTModel.from_pretrained(
        args.pretrained_path, model_name=args.model_name_or_path, device=args.device
    )
    model.to(args.device)
    model.eval()

    def form_utterance(
        turns,
        max_length: Optional[int] = None,
        user_token: Optional[str] = None,
        system_token: Optional[str] = None,
    ):
        cur_length, turn_idx = 0, len(turns) - 1
        utterances = []

        while turn_idx >= 0:
            turn = turns[turn_idx]
            special_token = user_token if turn["speaker"] == "USER" else system_token
            utterance = (f"{special_token} " if special_token else "") + turn["utterance"]

            tokenized = tokenizer.tokenize(utterance)
            if max_length is not None and cur_length + len(tokenized) > max_length:
                break
            else:
                utterances.append(utterance)
                cur_length += len(tokenized)
                turn_idx -= 1
        return " ".join(utterances[::-1])

    def form_input(
        turns,
        slot_description: str,
        answer: Optional[str] = None,
        max_length: Optional[int] = None,
        user_token: Optional[str] = None,
        system_token: Optional[str] = None,
    ) -> Tuple[str, BatchEncoding]:
        # [CLS] utterance [SEP] slot [SEP] answer [SEP]
        latter = (
            slot_description
            if answer is None
            else " ".join([slot_description, tokenizer.sep_token, answer])
        )
        latter_len = len(tokenizer.tokenize(latter))

        utterance = form_utterance(
            turns,
            max_length=(max_length - latter_len if max_length is not None else None),
            user_token=user_token,
            system_token=system_token,
        )

        return (utterance, latter), tokenizer([utterance], [latter], return_tensors="pt")

    # TODO: add max_length and user/system token to argparser and adapt to main
    def predict_single(
        dialogue, max_length: Optional[int] = None
    ) -> Tuple[str, List[Tuple[str, str]]]:
        service_slot_pairs = [
            (schema.service_by_name[service], slot)
            for service in dialogue["services"]
            for slot in schema.service_by_name[service].slots
        ]

        states = []
        for service, slot in service_slot_pairs:
            (utterance, _), encoded = form_input(
                dialogue["turns"], slot_description=slot.description, max_length=max_length
            )
            outputs = model(input_ids=encoded.input_ids.to(args.device))

            if outputs.slot_logits[0] > 0.5:
                state_value = None
                if slot.is_categorical:
                    scores = {}
                    for value in slot.possible_values:
                        (_, _), _encoded = form_input(
                            dialogue["turns"],
                            slot_description=slot.description,
                            answer=value,
                            max_length=max_length,
                        )
                        _outputs = model(_encoded.input_ids.to(args.device))
                        scores.update({value: _outputs.value_logits[0].item()})

                    state_value = max(scores.items(), key=lambda p: p[1])[0]
                else:
                    begin_index = outputs.begin_logits[0].argmax()
                    end_index = outputs.end_logits[0].argmax()

                    if (
                        encoded.token_to_sequence(0, begin_index) != 0
                        or encoded.token_to_sequence(0, end_index) != 0
                    ):
                        # if the answer is in the latter part, drop this
                        state_value = None
                    else:
                        begin_char_index = encoded.token_to_chars(begin_index).start
                        end_char_index = encoded.token_to_chars(end_index).end
                        state_value = utterance[begin_char_index : end_char_index + 1]

                if state_value is not None:
                    states.append((f"{service.name}-{slot.name}", state_value))

        return dialogue["dialogue_id"], states

    predictions = []

    for sample in tqdm(dataset, ncols=99, desc="Predicting"):
        prediction = predict_single(sample, max_length=model.max_position_embeddings - 10)
        predictions.append(prediction)

    IDs, states = zip(*sorted(predictions))
    states = list(
        map(
            lambda state: "None"
            if len(state) == 0
            else "|".join(f"{a}={b.strip()}" for a, b in state),
            states,
        )
    )
    pd.DataFrame({"id": IDs, "state": states}).to_csv(args.prediction_csv, index=False)

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

    # Prediction
    parser.add_argument("--prediction_csv", type=Path, default="prediction.csv")

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
