from dataclasses import dataclass
from typing import Dict
from transformers.file_utils import ModelOutput


@dataclass
class DSTEvaluateOutput(ModelOutput):
    metrics: Dict[str, float]
