from typing import Optional

from datasets.dataset import DSTDataset
from datasets.schema import Schema
from utils.logger import logger


class DSTDatasetForOverview(DSTDataset):
    def __init__(self, *args, schema: Optional[Schema] = None, **kwargs):
        super().__init__(*args, **kwargs)

        logger.info(f"Successfully loaded {len(self.data)} dialogues...")
        self.data = sorted(self.data, key=lambda d: d["dialogue_id"])

        self.schema = schema

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def overview(self):
        logger.info(f"Dataset size: {len(self)}")

        service_count = {service.name: 0 for service in self.schema.services}
        slot_count = {
            f"{service.name}--{slot.name}": 0
            for service in self.schema.services
            for slot in service.slots
        }

        for dialogue in self.data:
            for service in dialogue["services"]:
                service_count[service] += 1

            turn = dialogue["turns"][-1]
            for turn in filter(lambda turn: turn["speaker"] == "USER", dialogue["turns"]):
                for frame in turn["frames"]:
                    service = frame["service"]
                    for slot in frame["state"]["slot_values"]:
                        slot_count[f"{service}--{slot}"] += 1

        logger.info(
            "Service count:\n"
            + "\n".join(
                map(
                    lambda p: f"{p[0]:20s} [{p[1]}]",
                    sorted(service_count.items(), key=lambda p: p[1], reverse=True),
                )
            )
        )

        logger.info(
            "Slot count:\n"
            + "\n".join(
                map(
                    lambda p: f"{p[0]:40s} [{p[1]}]",
                    sorted(
                        filter(
                            lambda p: service_count[p[0].split("--")[0]] != 0, slot_count.items()
                        ),
                        key=lambda p: p[1],
                        reverse=True,
                    ),
                ),
            )
        )
