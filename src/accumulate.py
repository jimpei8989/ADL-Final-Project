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
    for data_path in sorted(args.input_dir.iterdir()):
        ret = []
        for dialogue in json.loads(data_path.read_bytes()):
            slot_values = defaultdict(dict)
            for i, t in enumerate(dialogue["turns"]):
                if t["speaker"] == "SYSTEM":
                    continue
                print(f"Turns {i} ------------------------------")
                print(t["utterance"])
                for j, f in enumerate(t["frames"]):
                    print(f["service"], f["state"]["slot_values"])
                #     slot_values[f["service"]].update(f["state"]["slot_values"])
                # dialogue["turns"][i]["frames"] = [
                #     {"service": k, "state": {"slot_values": dict(v)}}
                #     for k, v in slot_values.items()
                # ]
            exit()
            ret.append(dialogue)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json.dump(ret, (args.output_dir / data_path.name).open("w"), indent=1)


if __name__ == "__main__":
    args = get_args()

    generate_accumulate_jsons(args)