import torch
from torch import nn


class LSTMnetwork(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

        self.hidden = (
            torch.zeros(1, 1, self.hidden_size),
            torch.zeros(1, 1, self.hidden_size),
        )

    def forward(self, seq: list) -> float:
        """Forward pass in neural network.
        Args:
            seq (list): Input sequence
        Returns:
            float: Result of forward pass
        """
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, -1), self.hidden
        )
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]
