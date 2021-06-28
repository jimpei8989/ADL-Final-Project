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
    begin_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None

    # Useful in prediction
    def logits_by_index(self, index):
        return DSTModelOutput(
            slot_logits=self.slot_logits[index].cpu(),
            value_logits=self.value_logits[index].cpu(),
            begin_logits=self.begin_logits[index].cpu(),
            end_logits=self.end_logits[index].cpu(),
        )
