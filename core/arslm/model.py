import torch.nn as nn
from .arscell import ARSCell

class ARSLMModel(nn.Module):
    def __init__(self, arscell: ARSCell):
        super().__init__()
        self.cell = arscell

    def forward(self, h, x):
        return self.cell(h, x)
