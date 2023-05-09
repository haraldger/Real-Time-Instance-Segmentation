from math import ceil
import numpy as np
import torch
from torch import nn

class PredictionHeads(nn.Module):
    def __init__(self, in_channels=256, num_heads=5, num_anchors=3, num_classes=1, num_masks=32) -> None:
        super().__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_heads = num_heads

        self.conv_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        self.bbox_layers = nn.ModuleList()
        self.mask_layers = nn.ModuleList()
        for _ in range(self.num_heads):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            ))
            self.cls_layers.append(nn.Conv2d(in_channels, num_anchors*num_classes, kernel_size=3, stride=1, padding=1))
            self.bbox_layers.append(nn.Conv2d(in_channels, num_anchors*4, kernel_size=3, stride=1, padding=1))
            self.mask_layers.append(nn.Conv2d(in_channels, num_anchors*num_masks, kernel_size=3, stride=1, padding=1))

        self.sigmoid = nn.Sigmoid()  # For classification/confidence
        self.tanh = nn.Tanh()   # For mask

    def forward(self, x):
        # Output shapes:
        # cls: list of num_heads tensors of shape [N, num_anchors*num_classes, H, W]
        # bbox: list of num_heads tensors of shape [N, num_anchors*4, H, W]
        # mask_coefficients: list of num_heads tensors of shape [N, num_anchors*num_masks, H, W]

        # x: [P3, P4, P5, P6, P7]
        assert len(x) == self.num_heads

        out = list()
        for i, feature in enumerate(x):
            out.append(self.conv_layers[i](feature))

        cls = list()
        bbox = list()
        mask_coefficients = list()
        for i, feature in enumerate(out):
            cls.append(self.sigmoid(self.cls_layers[i](feature)))
            bbox.append(self.bbox_layers[i](feature))
            mask_coefficients.append(self.tanh(self.mask_layers[i](feature)))

        return cls, bbox, mask_coefficients
    

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
    cls, bbox, mask_coefficients = model(x)

    print(f"len(cls): {len(cls)}")
    assert len(cls) == 5
    print(f"len(bbox): {len(bbox)}")
    assert len(bbox) == 5
    print(f"len(mask_coefficients): {len(mask_coefficients)}")
    assert len(mask_coefficients) == 5

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
    cls, bbox, mask_coefficients = model(x)

    for i in range(len(cls)):
        print(f"cls[{i}].shape: {cls[i].shape}")
        assert cls[i].shape == (2, 3, ceil(69/(2**i)), ceil(69/(2**i)))
        print(f"bbox[{i}].shape: {bbox[i].shape}")
        assert bbox[i].shape == (2, 3*4, ceil(69/(2**i)), ceil(69/(2**i)))
        print(f"mask_coefficients[{i}].shape: {mask_coefficients[i].shape}")
        assert mask_coefficients[i].shape == (2, 3*32, ceil(69/(2**i)), ceil(69/(2**i)))

    print("Test passed!")
    print()

def run_tests():
    test_network_and_outputs()
    test_output_shapes()

