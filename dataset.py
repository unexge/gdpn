from typing import Iterable, Tuple
import torch
from torch.utils.data import Dataset

from vocab import Vocab
from util import convert_name_to_training_tensor


def collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


class NamesDataset(Dataset):
    def __init__(self, vocab: Vocab, names: Iterable[str]):
        self.vocab = vocab
        self.names = names

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return convert_name_to_training_tensor(self.vocab, self.names[idx])