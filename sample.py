import sys
import torch
import numpy as np

import config
from model import CharRNN
from vocab import Vocab, SOS_TOKEN, EOS_TOKEN, load_vocab_from_chars
from util import convert_idx_to_one_hot


def sample(model: CharRNN, vocab: Vocab) -> str:
    model.eval()

    with torch.no_grad():
        hidden = None

        input = torch.from_numpy(
            np.array([
                convert_idx_to_one_hot(vocab, vocab.char_to_idx(SOS_TOKEN))
            ])).unsqueeze(0)
        input = input.float().to(config.device)

        pred = ''

        for _ in range(8):
            out, hidden = model(input, hidden)

            if out.view(-1).div(0.8).exp().sum() == 0:
                continue

            topi = torch.multinomial(out.view(-1).div(0.8).exp(), 1)[0]

            if topi.item() == vocab.char_to_idx(EOS_TOKEN):
                break

            pred += vocab.idx_to_char(topi.item())

            input = torch.from_numpy(
                np.array([convert_idx_to_one_hot(vocab,
                                                 topi.item())])).unsqueeze(0)
            input = input.float().to(config.device)

        return pred


def load_checkpoint(path: str):
    checkpoint = torch.load(path)

    vocab = load_vocab_from_chars(checkpoint['vocab_chars'])
    model_state_dict = checkpoint['model_state_dict']

    return model_state_dict, vocab


if __name__ == '__main__':
    state_dict, vocab = load_checkpoint(sys.argv[1])

    # TODO: load hyperparameters from checkpoint?
    model = CharRNN(len(vocab), 256, 2)
    model.load_state_dict(state_dict)
    model.to(config.device)

    for _ in range(25):
        print(sample(model, vocab))
