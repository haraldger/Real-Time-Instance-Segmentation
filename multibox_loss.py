import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class MultiboxLoss(nn.Module):
    def __init__(self, neg_pos_ratio=3, cls_coef=1, dst_coef=1.5, mask_coef=6.125):
        """
        neg_pos_ratio (int): maximum number of negative boxes to postive boxes
        dst_coef (float): coefficient for the localization loss
        cls_coef (float): coefficient for the classification loss
        mask_coef (float): coefficient for the mask loss
        """
        super(MultiboxLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.neg_pos_ratio = neg_pos_ratio
        self.cls_coef = cls_coef
        self.dst_coef = dst_coef
        self.mask_coef = mask_coef

        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, confidence, predicted_locations, predicted_masks, gt_labels, gt_locations, gt_masks, num_objects):
        """
        confidence (batch_size, top_k, 1): class predictions
        predicted_locations (batch_size, top_k, 4): predicted bounding box locations
        predicted_masks (batch_size, top_k, 138, 138): predicted masks
        gt_labels (batch_size, k, 1): ground truth labels
        gt_locations (batch_size, k, 4): ground truth bounding box locations
        gt_masks (batch_size, k, 138, 138): ground truth masks
        num_objects (batch_size, 1): number of objects in each image

        The first num_objects[1] out of k are the ground truth. The rest are filled with zeros.
        """
        
        # Match predictions with ground truth
        idx_pred, idx_gt = self.match_predictions(confidence, predicted_locations, gt_labels, gt_locations, num_objects)
        
        # Move to GPU
        idx_pred = idx_pred.to(self.device)
        idx_gt = idx_gt.to(self.device)

        cls_loss = 0
        dst_loss = 0
        mask_loss = 0
        for batch in range(confidence.shape[0]):
            # Filter out -1s
            filtered_idx_pred = idx_pred[batch][idx_pred[batch] != -1]
            filtered_idx_gt = idx_gt[batch][idx_gt[batch] != -1]

            # Compute losses
            cls_loss += self.cross_entropy(confidence[batch][filtered_idx_pred], gt_labels[batch][filtered_idx_gt])
            dst_loss += self.smooth_l1(predicted_locations[batch][filtered_idx_pred], gt_locations[batch][filtered_idx_gt])
            
            # Mask loss uses pixel-wise cross entropy
            mask_preds = predicted_masks[batch][filtered_idx_pred].view(-1, 1)      # (top_k*138*138, 1)
            # mask_gt = gt_masks[batch][filtered_idx_gt].view(-1, 1)                  # (top_k*138*138, 1)
            # mask_loss += self.cross_entropy(mask_preds, mask_gt)

        # Normalize losses
        cls_loss /= confidence.shape[0]
        dst_loss /= confidence.shape[0]
        # mask_loss /= confidence.shape[0]

        loss = self.cls_coef * cls_loss + self.dst_coef * dst_loss + self.mask_coef * mask_loss
        return loss


        
    def match_predictions(self, cls_preds, bbox_preds, gt_labels, gt_bboxes, num_objects):
        """
        cls_preds (batch_size, top_k, 1): class predictions
        bbox_preds (batch_size, top_k, 4): predicted bounding box locations
        gt_labels (batch_size, k, 1): ground truth labels
        gt_bboxes (batch_size, k, 4): ground truth bounding box locations
        """

        # Move to CPU
        cls_preds = cls_preds.detach().cpu()
        bbox_preds = bbox_preds.detach().cpu()
        gt_bboxes = gt_bboxes.detach().cpu()
        gt_labels = gt_labels.detach().cpu()
        num_objects = num_objects.detach().cpu()

        idx_preds = torch.full((cls_preds.shape[0], cls_preds.shape[1]), fill_value=-1)
        idx_gts = torch.full((cls_preds.shape[0], cls_preds.shape[1]), fill_value=-1)
        for batch in range(cls_preds.shape[0]):
            # Flatten batches
            cls_preds_batch = cls_preds[batch].flatten(start_dim=0, end_dim=-1)     # (top_k)
            gt_labels_batch = gt_labels[batch].flatten(start_dim=0, end_dim=-1)     # (top_k)
            bbox_preds_batch = bbox_preds[batch].flatten(start_dim=0, end_dim=-2)   # (top_k, 4)
            gt_bboxes_batch = gt_bboxes[batch].flatten(start_dim=0, end_dim=-2)     # (top_k, 4)
            
            gt_labels_batch = gt_labels_batch[:num_objects[batch]]                 # Remove zeros -> (num_objects)
            gt_bboxes_batch = gt_bboxes_batch[:num_objects[batch]]                 # Remove zeros -> (num_objects, 4)

            # Construct cost matrix 
            cls_cost = torch.abs(cls_preds_batch.unsqueeze(0) - gt_labels_batch.unsqueeze(1))
            dst_cost = torch.cdist(bbox_preds_batch, gt_bboxes_batch, p=1).T

            # Hungarian algorithm
            cost = self.cls_coef * cls_cost + self.dst_coef * dst_cost
            idx_gt, idx_pred = linear_sum_assignment(cost)      # ndarray
            
            # Add batch dimension
            idx_pred = torch.tensor(idx_pred)
            idx_gt = torch.tensor(idx_gt)

            # Add to idx_preds
            idx_preds[batch][:len(idx_pred)] = idx_pred
            idx_gts[batch][:len(idx_gt)] = idx_gt
            
        return idx_preds, idx_gts      # (batch_size, top_k), padded with -1 after the first num_objects elements
    


# Tests

def setup_data():
    pred_cls = torch.tensor([[[0.95],   
                                [0.05],
                                [0.05],
                                [0.25], 
                                [0.25]],
                                [[0.05],
                                [0.95],
                                [0.05],
                                [0.25],
                                [0.25]]])
    pred_bbox = torch.tensor([[[0.1, 0.1, 0.2, 0.2],
                                [0.1, 0.1, 0.2, 0.2],
                                [0.1, 0.1, 0.2, 0.2],
                                [0.1, 0.1, 0.2, 0.2],
                                [0.1, 0.1, 0.2, 0.2]],
                                [[0.1, 0.1, 0.2, 0.2],
                                [0.1, 0.1, 0.2, 0.2],
                                [0.1, 0.1, 0.2, 0.2],
                                [0.1, 0.1, 0.2, 0.2],
                                [0.1, 0.1, 0.2, 0.2]]])
    pred_mask = torch.rand((2, 5, 138, 138))


    gt_label = torch.tensor([[[1.],
                                [1.],
                                [1.],
                                [0.],
                                [0.]],
                                [[1.],
                                [1.],
                                [0.],
                                [0.],
                                [0.]]])

    gt_bbox = torch.tensor([[[0.11, 0.11, 0.21, 0.21],
                                [0.11, 0.11, 0.21, 0.21],
                                [0.11, 0.11, 0.21, 0.21],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]],
                                [[0.1, 0.1, 0.2, 0.2],
                                [0.1, 0.1, 0.2, 0.2],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]]])

    gt_masks = torch.rand((2, 5, 138, 138))
    gt_masks[0, 3:, :, :] = 0

    num_objects = torch.tensor([[3], [2]])

    return pred_cls, pred_bbox, pred_mask, gt_label, gt_bbox, gt_masks, num_objects

def test_loss_forward():
    print("Running MultiboxLoss forward test...")
    pred_cls, pred_bbox, pred_mask, gt_label, gt_bbox, gt_masks, num_objects = setup_data()
    loss = MultiboxLoss()
    loss_value = loss(pred_cls, pred_bbox, pred_mask, gt_label, gt_bbox, gt_masks, num_objects)
    print(loss_value)
    print()

def run_tests():
    test_loss_forward()
