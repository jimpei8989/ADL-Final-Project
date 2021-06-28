from argparse import Namespace
import torch
import numpy as np
import random
from typing import Dict, Any


from transformers import PreTrainedTokenizerBase, trainer_utils


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    trainer_utils.set_seed(seed)


def get_dataset_kwargs(args: Namespace, max_length: int = 512) -> Dict[str, Any]:
    dataset_kwargs = {}
    dataset_kwargs["user_token"] = args.user_token
    dataset_kwargs["system_token"] = args.system_token
    dataset_kwargs["strategy"] = args.strategy
    dataset_kwargs["last_user_turn_only"] = args.last_user_turn_only
    dataset_kwargs["reserved_for_latter"] = args.reserved_for_latter
    dataset_kwargs["overlap_turns"] = args.overlap_turns
    dataset_kwargs["ensure_user_on_both_ends"] = not args.no_ensure_user_on_both_ends
    dataset_kwargs["max_seq_length"] = max_length

    return dataset_kwargs


def add_tokens(
    tokenizer: PreTrainedTokenizerBase, user_token: str, system_token: str
) -> PreTrainedTokenizerBase:
    def special_token_check(token: str, tokenizer: PreTrainedTokenizerBase):
        if token is not None:
            ids = tokenizer.convert_tokens_to_ids([token])
            if len(ids) == 1 and ids[0] != tokenizer.unk_token_id:
                return True
            return False
        return True

    assert special_token_check(user_token, tokenizer)
    assert special_token_check(system_token, tokenizer)
    if user_token is not None:
        tokenizer.add_special_tokens({"additional_special_tokens": [user_token]})
    if system_token is not None:
        tokenizer.add_special_tokens({"additional_special_tokens": [system_token]})

    return tokenizer
