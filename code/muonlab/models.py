"""Models for the Muon project."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully connected network, e.g. widths=[128, 256, 256, 256, 16].

    `hidden_matrices()` returns the weights between hidden layers (every layer except the
    first and the last). These get the Muon family; the input layer, output layer and all
    biases are trained with AdamW, following Jordan (2024).
    """

    def __init__(self, widths: list[int], activation: str = "tanh"):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))
        self.activation = {"tanh": torch.tanh, "relu": torch.relu}[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

    def hidden_matrices(self) -> list[nn.Parameter]:
        return [layer.weight for layer in self.layers[1:-1]]
