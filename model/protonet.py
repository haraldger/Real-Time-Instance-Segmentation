import torch.nn.functional as F
import numpy as np
import torch
from torch import nn

class Protonet(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=256, proto_dim=138, num_prototypes=32):
        super(Protonet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.proto_dim = proto_dim
        self.num_prototypes = num_prototypes

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(hidden_channels)
        self.conv5 = nn.Conv2d(hidden_channels, num_prototypes, kernel_size=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x))) 
        x = self.relu(self.bn3(self.conv3(x)))
        x = F.interpolate(x, size=(self.proto_dim, self.proto_dim), mode='bilinear', align_corners=False)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.conv5(x))
        return x

#testing
def test_protonet_forward():
    x = torch.randn(2, 256, 69, 69)
    model = Protonet()
    y = model(x)

    print(f'Protonet output shape: {y.shape}')
    assert y.shape == (2, 32, 138, 138)

    print('Test passed!')

def run_tests():
     test_protonet_forward()
