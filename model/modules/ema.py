from typing import Dict
import torch as t
import torch.nn as nn


class EMA(nn.Module):

    mu: float
    shadow: Dict[str, t.Tensor]

    def __init__(self, mu: float):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow: Dict[str, t.Tensor] = {}

    def register(self, name: str, val: t.Tensor) -> None:
        self.shadow[name] = val.clone()

    def forward(self, name: str, x: t.Tensor) -> t.Tensor:
        assert name in self.shadow
        new_average: t.Tensor = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average
