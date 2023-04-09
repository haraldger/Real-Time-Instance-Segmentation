import numpy as np
import torch
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x
    
class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return None
    
def resnet18():
    return None

def resnet50():
    return None

def resnet101():
    return None