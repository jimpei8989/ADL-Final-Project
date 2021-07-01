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
    parser.add_argument("--schema", "-s", type=Path, default="dataset/data/schema.json")
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
                    state[f"{frame['service'].lower()}-{k.lower()}"] = (
                        v[0].replace(",", "_").lower()
                    )
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

    is_categorical = {
        f"{schema['service_name']}-{slot['name']}".lower()
        for schema in json.loads(args.schema.read_bytes())
        for slot in schema["slots"]
        if slot["is_categorical"]
    }
    # print(len(labels), len(preds))
    # assert len(labels) == len(preds)

    correct = 0
    result_rec = {}
    missing, surplus, WA_categorical, WA_span = 0, 0, 0, 0
    AC_categorical, AC_span = 0, 0
    slot_TP, slot_FP, slot_FN = 0, 0, 0
    for dialogue_id in labels:
        if dialogue_id not in preds:
            print(f"dialogue_id {dialogue_id} not appear in prediction")
        else:
            tmp = defaultdict(dict)
            for k in labels[dialogue_id]:
                if k not in preds[dialogue_id]:
                    tmp["missing"][k] = labels[dialogue_id][k]
                    missing += 1
                    slot_FN += 1
                elif labels[dialogue_id][k] != preds[dialogue_id][k]:
                    tmp["WA"][k] = {
                        "ground truth": labels[dialogue_id][k],
                        "prediction": preds[dialogue_id][k],
                    }
                    if k in is_categorical:
                        WA_categorical += 1
                    WA_span += 1
                    slot_TP += 1
                else:
                    if k in is_categorical:
                        AC_categorical += 1
                    AC_span += 1
                    slot_TP += 1
            tmp["surplus"] = {
                k_pred: preds[dialogue_id][k_pred]
                for k_pred in preds[dialogue_id]
                if k_pred not in labels[dialogue_id]
            }

            surplus += len(tmp["surplus"])
            slot_FP += len(tmp["surplus"])
            if len(tmp["surplus"]) == 0:
                del tmp["surplus"]
            if len(tmp) == 0:
                correct += 1
            result_rec[dialogue_id] = tmp
    print(f"Accuracy = {correct/len(labels)} ; {correct} / {len(labels)}")
    print(f"missing = {missing}, surplus = {surplus}")
    print(
        f"Categorical correct / wrong = {AC_categorical}, {WA_categorical}; Span correct / wrong = {AC_span}, {WA_span}"
    )
    print(
        f"Slot precision = {slot_TP / (slot_TP + slot_FP)}, recall = {slot_TP / (slot_TP + slot_FN)}"
    )
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
