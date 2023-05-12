import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import prediction_heads, resnet_backbone, fpn, protonet
from utils import fast_nms

class Yolact(nn.Module):
    def __init__(self, backbone=None, num_classes=1, num_masks=32, top_k=100, mask_dim=512):
        super().__init__()

        self.num_classes = num_classes
        self.num_masks = num_masks
        self.top_k = top_k
        self.mask_dim = 512

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
        sort_idx = sort_idx[:, :self.top_k]    # Top 100 predictions
        cls = torch.gather(cls, 1, sort_idx.expand(-1, -1, cls.shape[-1]))
        reg = torch.gather(reg, 1, sort_idx.expand(-1, -1, reg.shape[-1]))
        mask_coefficients = torch.gather(mask_coefficients, 1, sort_idx.expand(-1, -1, mask_coefficients.shape[-1]))

        bboxes, classes, coefficients, masked_columns = fast_nms.batched_fnms(reg, cls, mask_coefficients, threshold=0.75)


        # ProtoNet
        proto_out = self.protonet(p_outputs[0])
        proto_out = proto_out.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, self.num_masks)

        # Matrix multiplication of coefficients and proto_out -> final masks
        coefficients = torch.transpose(coefficients, -1, -2)       
        masks = torch.matmul(proto_out, coefficients)
        masks = F.sigmoid(masks)
        masks = masks.view(x.shape[0], -1, self.mask_dim, self.mask_dim)

        return bboxes, classes, masks, masked_columns
    

    def evaluate(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)      # Add batch dimension, even if batch size is 1

        bboxes, classes, masks, columns_to_keep = self.forward(x)
        """
        bboxes: tensor of shape (batch_size, num_bboxes, 4)
        classes: tensor of shape (batch_size, num_bboxes) - contains class confidence scores
        masks: tensor of shape (batch_size, num_masks, 138, 138)
        """

        # Crop masks with predicted bounding boxes
        cropped_masks = []
        for i in range(len(bboxes)):
            cropped_masks.append([])
            for j in range(len(bboxes[i])):
                bbox = bboxes[i][j]
                mask = masks[i][j]
                cropped_mask = torch.zeros(mask.shape)
                cropped_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                cropped_masks[i].append(cropped_mask)

        return bboxes, classes, cropped_masks, columns_to_keep

        


    
# testing

def test_yolact_forward():
    print("Running Yolact forward test...")

    x = torch.randn(2, 3, 512, 512)
    model = Yolact()
    bboxes, classes, masks, columns_to_keep = model.forward(x)

    print(f'bboxes.shape: {bboxes.shape}')
    assert bboxes.shape == (2, 100, 4)

    print(f'classes.shape: {classes.shape}')
    assert classes.shape == (2, 100, 1)

    print(f'masks.shape: {masks.shape}')
    assert masks.shape == (2, 100, 512, 512)

    print(f'columns_to_keep.shape: {columns_to_keep.shape}')
    assert columns_to_keep.shape == (2, 100)

    print("Yolact forward test successful!")


def run_tests():
    test_yolact_forward()