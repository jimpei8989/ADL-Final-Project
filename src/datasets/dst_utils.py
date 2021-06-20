from pathlib import Path
from typing import Callable, Dict, Optional

from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from datasets.schema import Schema
from datasets.dataset_dst_for_slot import DSTDatasetForDSTForSlot
from datasets.dataset_dst_for_categorical import DSTDatasetForDSTForCategorical
from datasets.dataset_dst_for_span import DSTDatasetForDSTForSpan
from datasets.mt_dataloader import MTDataLoader

from utils.logger import logger


def load_dst_dataloader(
    data_dir: Path,
    schema_json: Path,
    tokenizer: PreTrainedTokenizerBase,
    dataloader_cls: Callable[[Dataset], DataLoader],
    dataset_kwargs: Optional[Dict] = None,
    for_slot_kwargs: Optional[Dict] = None,
    for_categorical_kwargs: Optional[Dict] = None,
    for_span_kwargs: Optional[Dict] = None,
):
    schema = Schema.load_json(schema_json)

    logger.info("Loading DSTDatasetForDSTForSlot")
    slot_dataset = DSTDatasetForDSTForSlot(
        data_dir,
        schema=schema,
        tokenizer=tokenizer,
        **(dataset_kwargs if dataset_kwargs is not None else {}),
        **(for_slot_kwargs if for_slot_kwargs is not None else {}),
    )

    logger.info("Loading DSTDatasetForDSTForCategorical")
    categorical_dataset = DSTDatasetForDSTForCategorical(
        data_dir,
        schema=schema,
        tokenizer=tokenizer,
        **(dataset_kwargs if dataset_kwargs is not None else {}),
        **(for_categorical_kwargs if for_categorical_kwargs is not None else {}),
    )

    logger.info("Loading DSTDatasetForDSTForSpan")
    span_dataset = DSTDatasetForDSTForSpan(
        data_dir,
        schema=schema,
        tokenizer=tokenizer,
        **(dataset_kwargs if dataset_kwargs is not None else {}),
        **(for_span_kwargs if for_span_kwargs is not None else {}),
    )

    dataloaders = [
        dataloader_cls(dataset) for dataset in [slot_dataset, categorical_dataset, span_dataset]
    ]

    mt_dataloader = MTDataLoader(*dataloaders)

    return mt_dataloader


if __name__ == "__main__":
    from datasets.dst_collator import DSTCollator
    from utils.tqdmm import tqdmm

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dst_collator = DSTCollator(pad_value=tokenizer.pad_token_id)

    def to_dataloader(dataset):
        return DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=dst_collator,
        )

    dataloader = load_dst_dataloader(
        data_dir=Path("../dataset/data-0614/train"),
        schema_json=Path("../dataset/data/schema.json"),
        tokenizer=tokenizer,
        dataloader_cls=to_dataloader,
        dataset_kwargs={"system_token": tokenizer.sep_token, "user_token": tokenizer.sep_token},
    )

    for i, batch in enumerate(tqdmm(dataloader, desc="Iterating through the dataloader")):
        if i < 3:
            print(f"\nBatch {i}: ", batch)
