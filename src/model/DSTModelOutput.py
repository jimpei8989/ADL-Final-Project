from dataclasses import dataclass
from typing import Optional
from transformers.file_utils import ModelOutput
import torch


@dataclass
class DSTModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    slot_loss: Optional[torch.FloatTensor] = None
    value_loss: Optional[torch.FloatTensor] = None
    span_loss: Optional[torch.FloatTensor] = None
    slot_logits: torch.FloatTensor = None
    value_logits: torch.FloatTensor = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
