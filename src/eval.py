from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
from collections import defaultdict
import pandas as pd


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--data_dir", "-g", type=Path, required=True)
    parser.add_argument("--pred_file", "-p", type=Path)
    parser.add_argument("--first_only", action="store_true")

    args = parser.parse_args()

    return args


def check_OR(labels):
    withOR = set()
    for ids, states in labels.items():
        for k, v in states.items():
            if "|" in v:
                withOR.add(f"{ids}-{k}")


def eval(args):
    labels = {}
    for data_path in args.data_dir.iterdir():
        for dialogue in json.loads(data_path.read_bytes()):
            state = defaultdict(str)
            for frame in dialogue["turns"][-2]["frames"]:
                for k, v in frame["state"]["slot_values"].items():
                    state[f"{frame['service']}-{k}"] = v[0]
            labels[dialogue["dialogue_id"]] = dict(state)
    # json.dump(labels, Path("dev_gt_forview.json").open("w"), indent=2)

    df = pd.read_csv(args.pred_file)
    preds = {}
    for _, row in df.iterrows():
        if row["state"] != "None":
            try:
                preds[row["id"]] = {
                    kv.split("=")[0]: kv.split("=")[1] for kv in row["state"].split("|")
                }
            except:
                preds[row["id"]] = {"Error": "contain |"}
                print(row["state"])
        else:
            preds[row["id"]] = {}

    assert len(labels) == len(preds)
    correct = 0
    for dialogue_id in labels:
        if (
            len(labels[dialogue_id]) == len(preds[dialogue_id])
            and labels[dialogue_id] == preds[dialogue_id]
        ):
            correct += 1
    print(f"Accuracy = {correct/len(labels)} ; {correct} / {len(labels)}")


if __name__ == "__main__":
    args = get_args()

    eval(args)
