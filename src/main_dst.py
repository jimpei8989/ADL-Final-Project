from argparse import ArgumentParser, Namespace
from pathlib import Path

from datasets.dataset_dst import DSTDatasetForDST
from datasets.schema import Schema
from utils.logger import logger


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default="dataset/data/")
    return parser.parse_args()


def main(args):
    logger.info(args)

    schema = Schema.load_json(args.dataset_dir / "schema.json")
    logger.info(
        "Schema successfully loaded.\n"
        f"Possible services: {[s.name for s in schema]}\n"
        f"Sample schema: {schema.services[0]}"
    )


if __name__ == "__main__":
    main(parse_args())
