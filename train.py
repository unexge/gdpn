import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from model import CharRNN
from vocab import build_vocab_from_list
from dataset import NamesDataset, collate


def train(model: CharRNN, optimizer: optim.Optimizer, criterion, inputs,
          targets):
    model.train()
    optimizer.zero_grad()

    hidden = None

    total_loss = 0

    for i in range(inputs.shape[0]):
        output, hidden = model(inputs[i].unsqueeze(0).unsqueeze(0).float(),
                               hidden)
        loss = criterion(output.squeeze(0), targets[i].unsqueeze(0).long())
        total_loss += loss

    total_loss.backward()
    optimizer.step()

    return output, total_loss.item() / inputs.shape[0]


def save_checkpoint(path, model, vocab):
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'vocab_chars': vocab._chars
        }, path)


if __name__ == '__main__':
    names = []
    with open('./data/names.txt') as f:
        names = f.read().split('\n')

    vocab = build_vocab_from_list(names)

    names_dataset = NamesDataset(vocab, names)
    dataloader = DataLoader(names_dataset,
                            batch_size=32,
                            shuffle=True,
                            collate_fn=collate)

    model = CharRNN(len(vocab), 256, 2)
    model.to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epoch = 8

    total_loss_item = 0
    total_loss = 0

    total_mini_batch = len(dataloader)

    for e in range(n_epoch):
        for i, example in enumerate(dataloader):
            for j in range(len(example[0])):
                input, target = example[0][j], example[1][j]
                output, loss = train(model, optimizer, criterion, input,
                                     target)
                total_loss += loss
                total_loss_item += 1

        loss_str = '{:.2f}'.format(total_loss / total_loss_item)

        checkpoint_filename = './data/model/checkpoint-{}-{}.tar'.format(
            e,
            str(loss_str).replace('.', '_'))
        save_checkpoint(
            checkpoint_filename,
            model,
            vocab,
        )
        print('(EPOCH {}) loss {} checkpoint saved to {}'.format(
            e, loss_str, checkpoint_filename))
