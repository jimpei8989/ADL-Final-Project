from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import numpy as np


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--pred_file", type=Path)
    parser.add_argument("--first_only", action="store_true")

    args = parser.parse_args()

    return args


def analyze(data: np.ndarray, name: str) -> None:
    print(f"analyze {name} with {len(data)} data")
    for quantile in [50, 75, 95, 97.5, 99, 99.5]:
        print(f"{quantile}% quantile of data is {np.percentile(data, quantile)}")


if __name__ == "__main__":
    args = get_args()

    labels = {
        dialogue["dialogue_id"]: dialogue["turns"][-2]["state"]
        for data_path in args.data_dir.iterdir()
        for dialogue in json.loads(data_path.read_bytes())
    }

    with open(args.pred_file, "r") as f:
        preds = {line.split(',')[0]: line.split(',')[1] for line in f}

    assert len(labels) == len(preds)
    for dialogue_id in labels:
        label = labels[dialogue_id]
        pred = preds[dialogue_id]
        pred = {for pair in pred.split('|')}


    turn_num = np.array(turn_num)
    total_utterance_len = np.array(total_utterance_len)
    for max_length in range(480, 520, 10):
        print(
            f"there are {(total_utterance_len > max_length).mean()} of {len(total_utterance_len)} dialogue more than {max_length}"
        )

    analyze(turn_num, "turn num")
    analyze(total_utterance_len, "total_utterance_len")
