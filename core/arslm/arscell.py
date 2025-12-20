from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class ARSCell(nn.Module, ABC):
    """
    Public ARSCell interface (safe).
    Proprietary internals are abstracted.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    @abstractmethod
    def forward(self, h_prev, x):
        pass
