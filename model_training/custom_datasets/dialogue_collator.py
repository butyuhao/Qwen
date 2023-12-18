import random
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from model_training.custom_datasets.formatting import QA_SPECIAL_TOKENS, format_pairs, format_system_prefix
from torch.nn import functional as F
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase, TruncationStrategy
import transformers

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import torch
import transformers

from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class DialogueDataCollator:
    """
    Expects a list of texts corresponding to a sequence of [question, answer, question, answer, ...] pairs.
    """

    tokenizer: PreTrainedTokenizerBase
    # padding: Union[bool, str, PaddingStrategy] = True
    max_len: Optional[int] = None
    use_system_prefix: bool = False
    system_prefix: str = None


    def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = "You are a helpful assistant."
    ) -> Dict:
        roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

        im_start = tokenizer.im_start_id
        im_end = tokenizer.im_end_id
        nl_tokens = tokenizer('\n').input_ids
        _system = tokenizer('system').input_ids + nl_tokens
        _user = tokenizer('user').input_ids + nl_tokens
        _assistant = tokenizer('assistant').input_ids + nl_tokens

        # Apply prompt templates
        input_ids, targets = [], []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != roles["user"]:
                source = source[1:]

            input_id, target = [], []
            system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
            input_id += system
            target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
            assert len(input_id) == len(target)
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                _input_id = tokenizer(role).input_ids + nl_tokens + \
                    tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
                input_id += _input_id
                if role == '<|im_start|>user':
                    _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
                elif role == '<|im_start|>assistant':
                    _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                        _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
                else:
                    raise NotImplementedError
                target += _target
            assert len(input_id) == len(target)
            input_ids.append(input_id[:max_len])
            targets.append(target[:max_len])
        
        # Pad to batch max_len
        batch_pad_max_len = max_len
        batch_max_length = max(len(i) for i in input_ids)
        batch_pad_max_len = min(batch_pad_max_len, batch_max_length)

        input_ids = [i + [tokenizer.pad_token_id] * (batch_pad_max_len - len(i)) for i in input_ids]
        targets = [t + [IGNORE_TOKEN_ID] * (batch_pad_max_len - len(t)) for t in targets]

        input_ids = torch.tensor(input_ids, dtype=torch.int)
        targets = torch.tensor(targets, dtype=torch.int)

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )


    def __call__(self, features):
        # import ipdb
        # ipdb.set_trace()

        for messages in features:
            if isinstance(messages,tuple) and len(messages)==3 and messages[-1]=="<|H|>":
                dialogue = messages[0]
                system_prefix = messages[1]
            else:
                dialogue = messages
                system_prefix = self.system_prefix

            dialogue = [{"from": "user", "value": dialogue[i]} if i%2==0 else {"from": "assistant", "value": dialogue[i]} for i, t in enumerate(dialogue)]
            batch_data = self.preprocess(dialogue, system_prefix)


        return batch_data
