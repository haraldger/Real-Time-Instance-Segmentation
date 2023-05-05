import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import prediction_heads, resnet_backbone, fpn, protonet

class Yolact(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()

        if backbone is None:
            self.backbone = resnet_backbone.resnet50()
        else:
            self.backbone = backbone

        self.fpn = fpn.FPN()
        self.prediction_heads = prediction_heads.PredictionHeads()
        #self.protonet = protonet.Protonet()

        self.bbox_format = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32)

    def forward(self, x):
        raise NotImplementedError

    def postprocess(self, cls_logits, reg_logits, mask_logits, anchors, h, w, top_k=100, score_threshold=0.05, nms_threshold=0.5):
        # cls_logits: list of num_heads tensors of shape [N, num_anchors*num_classes, H, W]
        # reg_logits: list of num_heads tensors of shape [N, num_anchors*4, H, W]
        # mask_logits: list of num_heads tensors of shape [N, num_anchors*num_masks, H, W]
        # anchors: list of num_heads tensors of shape [num_anchors, 4]
        # h: list of num_heads tensors of shape [N]
        # w: list of num_heads tensors of shape [N]
        # top_k: int
        # score_threshold: float
        # nms_threshold: float

        # Output shapes:
        # bboxes
        raise NotImplementedError