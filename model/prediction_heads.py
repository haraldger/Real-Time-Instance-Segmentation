from math import ceil
import numpy as np
import torch
from torch import nn

class PredictionHeads(nn.Module):
    def __init__(self, in_channels, num_heads=5, num_anchors=3, num_classes=2, num_masks=32) -> None:
        super().__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_heads = num_heads

        self.conv_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        self.reg_layers = nn.ModuleList()
        self.mask_layers = nn.ModuleList()
        for _ in range(self.num_heads):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ))
            self.cls_layers.append(nn.Conv2d(in_channels, num_anchors*num_classes, kernel_size=3, stride=1, padding=1))
            self.reg_layers.append(nn.Conv2d(in_channels, num_anchors*4, kernel_size=3, stride=1, padding=1))
            self.mask_layers.append(nn.Conv2d(in_channels, num_anchors*num_masks, kernel_size=3, stride=1, padding=1))

        self.softmax = nn.Softmax(dim=1)  # For classification
        self.tanh = nn.Tanh()   # For mask

    def forward(self, x):
        # Output shapes:
        # cls_logits: list of num_heads tensors of shape [N, num_anchors*num_classes, H, W]
        # reg_logits: list of num_heads tensors of shape [N, num_anchors*4, H, W]
        # mask_logits: list of num_heads tensors of shape [N, num_anchors*num_masks, H, W]

        # x: [P3, P4, P5, P6, P7]
        assert len(x) == self.num_heads

        out = list()
        for i, feature in enumerate(x):
            out.append(self.conv_layers[i](feature))

        cls_logits = list()
        reg_logits = list()
        mask_logits = list()
        for i, feature in enumerate(out):
            cls_logits.append(self.softmax(self.cls_layers[i](feature)))
            reg_logits.append(self.reg_layers[i](feature))
            mask_logits.append(self.tanh(self.mask_layers[i](feature)))

        return cls_logits, reg_logits, mask_logits
    

# Test PredictionHeads module

def test_network_and_outputs():
    print("Testing PredictionHeads module...")

    p3 = torch.randn((2, 256, 69, 69))
    p4 = torch.randn((2, 256, 35, 35))
    p5 = torch.randn((2, 256, 18, 18))
    p6 = torch.randn((2, 256, 9, 9))
    p7 = torch.randn((2, 256, 5, 5))
    x = [p3, p4, p5, p6, p7]

    model = PredictionHeads(256)
    cls_logits, reg_logits, mask_logits = model(x)

    print(f"len(cls_logits): {len(cls_logits)}")
    assert len(cls_logits) == 5
    print(f"len(reg_logits): {len(reg_logits)}")
    assert len(reg_logits) == 5
    print(f"len(mask_logits): {len(mask_logits)}")
    assert len(mask_logits) == 5

    print("Test passed!")

def test_output_shapes():
    print("Testing output shapes of PredictionHeads module...")

    p3 = torch.randn((2, 256, 69, 69))
    p4 = torch.randn((2, 256, 35, 35))
    p5 = torch.randn((2, 256, 18, 18))
    p6 = torch.randn((2, 256, 9, 9))
    p7 = torch.randn((2, 256, 5, 5))
    x = [p3, p4, p5, p6, p7]

    model = PredictionHeads(256)
    cls_logits, reg_logits, mask_logits = model(x)

    for i in range(len(cls_logits)):
        print(f"cls_logits[{i}].shape: {cls_logits[i].shape}")
        assert cls_logits[i].shape == (2, 3*2, ceil(69/(2**i)), ceil(69/(2**i)))
        print(f"reg_logits[{i}].shape: {reg_logits[i].shape}")
        assert reg_logits[i].shape == (2, 3*4, ceil(69/(2**i)), ceil(69/(2**i)))
        print(f"mask_logits[{i}].shape: {mask_logits[i].shape}")
        assert mask_logits[i].shape == (2, 3*32, ceil(69/(2**i)), ceil(69/(2**i)))

    print("Test passed!")

def run_tests():
    test_network_and_outputs()
    test_output_shapes()

