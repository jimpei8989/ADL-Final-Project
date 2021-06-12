from argparse import ArgumentParser, Namespace
from pathlib import Path

from datasets.dataset_dst import DSTDatasetForDST
from datasets.schema import Schema
from utils.logger import logger


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--train_data", type=Path, default="dataset/data/train/")
    parser.add_argument("--schema_json", type=Path, default="dataset/data/schema.json")
    return parser.parse_args()


def main(args):
    logger.info(args)

    schema = Schema.load_json(args.schema_json)
    logger.info(
        "Schema successfully loaded.\n"
        f"Possible services: {[s.name for s in schema]}\n"
        f"Sample schema: {schema.services[0]}"
    )

    dataset = DSTDatasetForDST(json_dir=args.train_data, schema=schema)


if __name__ == "__main__":
    main(parse_args())
