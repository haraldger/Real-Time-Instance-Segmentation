import numpy as np
import torch
from torch import nn

class FPN(nn.Module):
    def __init__(self, in_channels_list=[128, 256, 512], out_channels=256) -> None:
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

        # C5: [N, 512, 16, 16]
        # C4: [N, 256, 32, 32]
        # C3: [N, 128, 64, 64]

        # P5: [N, 256, 16, 16]
        # P4: [N, 256, 32, 32]
        # P3: [N, 256, 64, 64]

        # P6: [N, 256, 8, 8]
        # P7: [N, 256, 4, 4]

        out = list()
        batch_size = inputs[0].shape[0]
        x = torch.zeros((batch_size, 256, 1, 1)).to(inputs[0].device)   # Initialize P5

        for i, feature in enumerate(inputs[::-1]):      # Reverse the list
            lateral_feature = self.lateral_convs[len(inputs) - i - 1](feature)
            h,w = lateral_feature.shape[-2:]
            x = lateral_feature + nn.functional.interpolate(x, size=(h,w), mode='nearest')
            out.insert(0, self.relu(self.output_convs[i](x)))   # Insert at the beginning

        out.append(self.relu(self.p6_conv(out[-1])))
        out.append(self.relu(self.p7_conv(out[-1])))

        return out    # [P3, P4, P5, P6, P7]



# Test FPN module

def test_network_and_shapes():
    x = torch.randn((2, 128, 64, 64))
    y = torch.randn((2, 256, 32, 32))
    z = torch.randn((2, 512, 16, 16))

    fpn = FPN()
    out = fpn([x,y,z])

    print(f'Length of output: {len(out)}')
    assert len(out) == 5

    print(f'Shape of P3: {out[0].shape}')
    assert out[0].shape == (2, 256, 64, 64)

    print(f'Shape of P4: {out[1].shape}')
    assert out[1].shape == (2, 256, 32, 32)

    print(f'Shape of P5: {out[2].shape}')
    assert out[2].shape == (2, 256, 16, 16)

    print(f'Shape of P6: {out[3].shape}')
    assert out[3].shape == (2, 256, 8, 8)

    print(f'Shape of P7: {out[4].shape}')
    assert out[4].shape == (2, 256, 4, 4)

    print('Test passed')

def run_tests():
    test_network_and_shapes()



        