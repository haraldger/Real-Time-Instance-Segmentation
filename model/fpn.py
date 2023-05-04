import numpy as np
import torch
from torch import nn

class FPN(nn.Module):
    def __init__(self, in_channels_list=[64, 128, 256, 512], out_channels=256) -> None:
        super().__init__()

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))

        self.p6_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.p7_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU()

    def forward(self, inputs: list) -> list:
        # inputs: [C3, C4, C5]

        # C5: [N, 512, 18, 18]
        # C4: [N, 256, 35, 35]
        # C3: [N, 128, 69, 69]
        # C2: [N, 64, 138, 138]     

        # P5: [N, 256, 18, 18]
        # P4: [N, 256, 35, 35]
        # P3: [N, 256, 69, 69]

        out = list()
        batch_size = inputs[0].shape[0]
        x = torch.zeros((batch_size, 256, 1, 1)).to(inputs[0].device)   # Initialize P5

        for i, feature in enumerate(inputs[::-1]):
            lateral_feature = self.lateral_convs[i](feature)
            h,w = lateral_feature.shape[-2:]
            x = lateral_feature + nn.functional.interpolate(x, size=(h,w), mode='nearest')
            out.append(self.relu(self.output_convs[i](x)))

        # Reverse the list
        out = out[::-1]    

        out.append(self.relu(self.p6_conv(out[-1])))
        out.append(self.relu(self.p7_conv(out[-1])))

        return out[::-1]    # [P3, P4, P5, P6, P7]




        