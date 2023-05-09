import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import prediction_heads, resnet_backbone, fpn, protonet

class Yolact(nn.Module):
    def __init__(self, backbone=None, num_classes=2, num_masks=32):
        super().__init__()

        self.num_classes = num_classes
        self.num_masks = num_masks

        # Network components
        if backbone is None:
            self.backbone = resnet_backbone.resnet50()
        else:
            self.backbone = backbone
        self.fpn = fpn.FPN()
        self.prediction_heads = prediction_heads.PredictionHeads(num_classes=self.num_classes, num_masks=self.num_masks)
        self.protonet = protonet.Protonet()

    def forward(self, x):
        # Backbone + FPN
        c_outputs = self.backbone(x)
        p_outputs = self.fpn(c_outputs)

        # Prediction heads
        cls_preds, reg_preds, mask_coefficients = self.prediction_heads(p_outputs)
        for idx in range(len(cls_preds)):
            cls_preds[idx] = cls_preds[idx].permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, self.num_classes)
            reg_preds[idx] = reg_preds[idx].permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 4)
            mask_coefficients[idx] = mask_coefficients[idx].permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, self.num_masks)
        
        cls = torch.cat(cls_preds, dim=1)
        reg = torch.cat(reg_preds, dim=1)
        mask_coefficients = torch.cat(mask_coefficients, dim=1)

        # Sort according to confidence
        _, sort_idx = torch.sort(cls, dim=1, descending=True)
        cls = torch.gather(cls, 1, sort_idx)
        reg = torch.gather(reg, 1, sort_idx)
        mask_coefficients = torch.gather(mask_coefficients, 1, sort_idx)

        

        # ProtoNet
        proto_out = self.protonet(p_outputs[0])

        return torch.randn(2, 256, 69, 69)
        


    
# testing

def test_yolact_forward():
    x = torch.randn(2, 3, 550, 550)
    model = Yolact()
    y = model(x)

    # TODO: Implement meaningful test with assertions

def run_tests():
    test_yolact_forward()