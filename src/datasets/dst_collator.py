from typing import Any, Dict, List

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate


class DSTCollator:
    ignored = {"type"}

    def __init__(self, pad_value) -> None:
        self.pad_value = pad_value

    def __call__(self, batches: List[Dict[str, Any]]) -> Dict[str, Any]:
        first = batches[0]

        ret = {}
        if "type" in first:
            ret.update({"type": first["type"]})

        for key in filter(lambda k: k not in self.ignored, first.keys()):
            if key == "input_ids":
                ret.update(
                    {
                        key: pad_sequence(
                            [b[key] for b in batches],
                            batch_first=True,
                            padding_value=self.pad_value,
                        )
                    }
                )
            elif key.startswith("_"):
                ret.update({key: [b[key] for b in batches]})
            else:
                ret.update({key: default_collate([b[key] for b in batches])})
        return ret
