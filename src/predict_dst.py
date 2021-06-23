from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import torch
import torch.nn.functional as nnf
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
        dialogue,
        max_length: Optional[int] = None,
        user_token: Optional[str] = None,
        system_token: Optional[str] = None,
    ) -> Tuple[str, List[Tuple[str, str]]]:
        service_slot_pairs = [
            (schema.service_by_name[service], slot)
            for service in dialogue["services"]
            for slot in schema.service_by_name[service].slots
        ]

        states = {}
        for service, slot in service_slot_pairs:
            for i in range(0, len(dialogue["turns"]), 2):
                (utterance, _), encoded = form_input(
                    dialogue["turns"][: i + 1],
                    slot_description=slot.description,
                    max_length=max_length,
                    user_token=user_token,
                    system_token=system_token,
                )
                outputs = model(input_ids=encoded.input_ids.to(args.device))

                if outputs.slot_logits[0] > 0:
                    state_value = None
                    if slot.is_categorical:
                        scores = {}
                        for value in slot.possible_values:
                            (_, _), _encoded = form_input(
                                dialogue["turns"],
                                slot_description=slot.description,
                                answer=value,
                                max_length=max_length,
                                user_token=user_token,
                                system_token=system_token,
                            )
                            _outputs = model(_encoded.input_ids.to(args.device))
                            scores.update({value: _outputs.value_logits[0].item()})

                        state_value = max(scores.items(), key=lambda p: p[1])[0]
                    else:
                        begin_scores = nnf.softmax(outputs.begin_logits[0], dim=0)
                        end_scores = nnf.softmax(outputs.end_logits[0], dim=0)

                        begin_index = begin_scores.argmax()
                        end_index = end_scores.argmax()

                        # Pair1: begin_index, _end_index
                        _end_scores = end_scores.clone()
                        _end_scores[:begin_index] = 0
                        _end_scores[begin_index + args.max_span_length :] = 0
                        _end_index = _end_scores.argmax()

                        pair1_score = begin_scores[begin_index] * end_scores[_end_index]
                        if (
                            encoded.token_to_sequence(0, begin_index) != 0
                            or encoded.token_to_sequence(0, _end_index) != 0
                        ):
                            pair1_score = 0

                        # Pair2: _begin_index, end_index
                        _begin_scores = begin_scores.clone()
                        _begin_scores[end_index + 1 :] = 0
                        _begin_scores[: max(1, end_index - args.max_span_length + 1)] = 0
                        _begin_index = _begin_scores.argmax()

                        pair2_score = begin_scores[_begin_index] * end_scores[end_index]
                        if (
                            encoded.token_to_sequence(0, _begin_index) != 0
                            or encoded.token_to_sequence(0, end_index) != 0
                        ):
                            pair2_score = 0

                        try:
                            if pair1_score == 0 and pair2_score == 0:
                                state_value = None
                            elif pair1_score > pair2_score:
                                begin_char_index = encoded.token_to_chars(begin_index).start
                                end_char_index = encoded.token_to_chars(_end_index).end
                                state_value = utterance[
                                    begin_char_index : end_char_index + 1
                                ].strip(STRIP_CHARS)
                            else:
                                begin_char_index = encoded.token_to_chars(_begin_index).start
                                end_char_index = encoded.token_to_chars(end_index).end
                                state_value = utterance[
                                    begin_char_index : end_char_index + 1
                                ].strip(STRIP_CHARS)
                        except TypeError:
                            state_value = None

                    if state_value is not None:
                        states[f"{service.name}-{slot.name}"] = state_value

        return dialogue["dialogue_id"], list(states.items())

    predictions = []

    for sample in tqdm(dataset, ncols=99, desc="Predicting"):
        prediction = predict_single(
            sample,
            max_length=model.max_position_embeddings - 10,
            user_token=args.user_token,
            system_token=args.system_token,
        )
        predictions.append(prediction)

    IDs, states = zip(*sorted(predictions))
    states = list(
        map(
            lambda state: "None" if len(state) == 0 else "|".join(f"{a}={b}" for a, b in state),
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
