from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
from collections import defaultdict


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--input_dir", "-i", type=Path, required=True)
    parser.add_argument("--output_dir", "-o", type=Path, required=True)

    args = parser.parse_args()

    return args


def generate_accumulate_jsons(args):
    for data_path in args.input_dir.iterdir():
        ret = []
        slot_values = defaultdict(dict)
        for dialogue in json.loads(data_path.read_bytes()):
            for i, t in enumerate(dialogue["turns"]):
                if t["speaker"] == "SYSTEM":
                    continue
                for j, f in enumerate(t["frames"]):
                    # print(f["state"]["slot_values"])
                    slot_values[f["service"]].update(f["state"]["slot_values"])
                    dialogue["turns"][i]["frames"][j]["state"][
                        "slot_values"
                    ] = slot_values[f["service"]]
            ret.append(dialogue)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json.dump(ret, open(args.output_dir / data_path.name, "w"))


if __name__ == "__main__":
    args = get_args()

    generate_accumulate_jsons(args)