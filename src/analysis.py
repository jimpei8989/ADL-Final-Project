from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import numpy as np


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=Path)

    args = parser.parse_args()

    return args


def analyze(data: np.ndarray, name: str) -> None:
    print(f"analyze {name} with {len(data)} data")
    for quantile in [50, 75, 95, 97.5, 99, 99.5]:
        print(f"{quantile}% quantile of data is {np.percentile(data, quantile)}")


if __name__ == "__main__":
    args = get_args()
    turn_num = []
    total_utterance_len = []
    for data_path in args.data_dir.iterdir():
        data = json.loads(data_path.read_bytes())
        for dialogue in data:
            utterances = [turn["utterance"] for turn in dialogue["turns"]]
            total_utterance = "".join(utterances)
            turn_num.append(len(utterances))
            total_utterance_len.append(len(total_utterance))

    turn_num = np.array(turn_num)
    total_utterance_len = np.array(total_utterance_len)
    for max_length in range(480, 520, 10):
        print(
            f"there are {(total_utterance_len > max_length).mean()} of {len(total_utterance_len)} dialogue more than {max_length}"
        )

    analyze(turn_num, "turn num")
    analyze(total_utterance_len, "total_utterance_len")
