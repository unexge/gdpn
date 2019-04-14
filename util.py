from typing import Tuple
import torch
import numpy as np

import config
from vocab import Vocab, SOS_TOKEN, EOS_TOKEN


def convert_idx_to_one_hot(vocab: Vocab, idx: int) -> np.ndarray:
    encoding = np.zeros((len(vocab)))
    encoding[idx] = 1

    return encoding


def convert_name_to_training_tensor(vocab: Vocab, name: str
                                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    idxs = list(map(vocab.char_to_idx, name))

    input = [vocab.char_to_idx(SOS_TOKEN)] + idxs
    target = idxs + [vocab.char_to_idx(EOS_TOKEN)]

    input = torch.from_numpy(
        np.array(
            list(
                map(lambda x, vocab=vocab: convert_idx_to_one_hot(vocab, x),
                    input))))
    target = torch.from_numpy(np.array(target))

    input = input.to(config.device)
    target = target.to(config.device)

    return input, target
