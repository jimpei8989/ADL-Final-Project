from pathlib import Path
from typing import Callable, Dict, Optional

from torch.utils.data import Dataset, DataLoader

from transformers import Trainer
from transformers import PreTrainedTokenizerBase

from datasets.dst_utils import load_dst_dataloader


class DSTTrainer(Trainer):
    def __init__(
        self,
        train_data_dir: Path,
        evaluate_data_dir: Path,
        schema_json: Path,
        tokenizer: PreTrainedTokenizerBase,
        dataloader_cls: Callable[[Dataset], DataLoader],
        dataset_kwargs: Optional[Dict] = None,
        for_slot_kwargs: Optional[Dict] = None,
        for_categorical_kwargs: Optional[Dict] = None,
        for_span_kwargs: Optional[Dict] = None,
        **kwargs
    ):
        self.train_dataloader = load_dst_dataloader(
            data_dir=train_data_dir,
            schema_json=schema_json,
            tokenizer=tokenizer,
            dataloader_cls=dataloader_cls,
            dataset_kwargs=dataset_kwargs,
            for_slot_kwargs=for_slot_kwargs,
            for_categorical_kwargs=for_categorical_kwargs,
            for_span_kwargs=for_span_kwargs,
        )
        self.evaluate_dataloader = load_dst_dataloader(
            data_dir=evaluate_data_dir,
            schema_json=schema_json,
            tokenizer=tokenizer,
            dataloader_cls=dataloader_cls,
            dataset_kwargs=dataset_kwargs,
            for_slot_kwargs=for_slot_kwargs,
            for_categorical_kwargs=for_categorical_kwargs,
            for_span_kwargs=for_span_kwargs,
        )
        super().__init__(**kwargs)

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_evaluate_dataloader(self):
        return self.evaluate_dataloader
