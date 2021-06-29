from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
from collections import defaultdict
import pandas as pd
import os


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--data_dir", "-g", type=Path, required=True)
    parser.add_argument("--pred_file", "-p", type=Path)
    parser.add_argument("--first_only", action="store_true")
    parser.add_argument("--save_for_view", action="store_true")

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

    if args.save_for_view:
        os.makedirs("forview/", exist_ok=True)
        json.dump(
            labels,
            Path("forview/dev_gt_forview.json").open("w"),
            indent=2,
            sort_keys=True,
        )

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

    if args.save_for_view:
        json.dump(
            preds,
            Path("forview/dev_pred_forview.json").open("w"),
            indent=2,
            sort_keys=True,
        )
    # print(len(labels), len(preds))
    # assert len(labels) == len(preds)

    correct = 0
    result_rec = {}
    missing, surplus, wrong_answer = 0, 0, 0
    for dialogue_id in labels:
        if dialogue_id not in preds:
            print(f"dialogue_id {dialogue_id} not appear in prediction")
        else:
            tmp = defaultdict(dict)
            for k in labels[dialogue_id]:
                if k not in preds[dialogue_id]:
                    tmp["missing"][k] = labels[dialogue_id][k]
                    missing += 1
                elif labels[dialogue_id][k] != preds[dialogue_id][k]:
                    tmp["WA"][k] = {
                        "ground truth": labels[dialogue_id][k],
                        "prediction": preds[dialogue_id][k],
                    }
                    wrong_answer += 1
            tmp["surplus"] = {
                k_pred: preds[dialogue_id][k_pred]
                for k_pred in preds[dialogue_id]
                if k_pred not in labels[dialogue_id]
            }

            surplus += len(tmp["surplus"])
            if len(tmp["surplus"]) == 0:
                del tmp["surplus"]
            if len(tmp) == 0:
                correct += 1
            result_rec[dialogue_id] = tmp
    print(f"Accuracy = {correct/len(labels)} ; {correct} / {len(labels)}")
    print(f"missing = {missing}, surplus = {surplus}, wrong answer = {wrong_answer}")
    if args.save_for_view:
        json.dump(
            result_rec,
            Path("forview/difference.json").open("w"),
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    args = get_args()

    eval(args)
