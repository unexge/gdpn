import torch
import torch.nn as nn

import config


class CharRNN(nn.Module):
    def __init__(self, n_chars, hidden_size, num_layers):
        super(CharRNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(n_chars,
                          hidden_size,
                          num_layers,
                          dropout=0.5,
                          batch_first=True)

        self.fc = nn.Linear(hidden_size, n_chars)

    def forward(self, input, hidden=None):
        out, hidden = self.gru(input, hidden)
        out = self.fc(out)

        return out, hidden
