from typing import Iterable

EOS_TOKEN = 'EOS'
SOS_TOKEN = 'SOS'


class Vocab:
    def __init__(self, chars: Iterable[str]):
        self._chars = chars
        self.__char_to_idx = {c: i for (i, c) in enumerate(chars, 2)}
        self.__char_to_idx[EOS_TOKEN] = 0
        self.__char_to_idx[SOS_TOKEN] = 1

        self.__idx_to_char = {i: c for c, i in self.__char_to_idx.items()}

    def __len__(self) -> int:
        return len(self.__char_to_idx)

    def char_to_idx(self, char: str) -> int:
        return self.__char_to_idx[char]

    def idx_to_char(self, idx: int) -> str:
        return self.__idx_to_char[idx]


def build_vocab_from_list(names: Iterable[str]) -> Vocab:
    uniq_chars = set(''.join(names))

    return Vocab(uniq_chars)


def load_vocab_from_chars(chars: Iterable[str]) -> Vocab:
    return Vocab(chars)
