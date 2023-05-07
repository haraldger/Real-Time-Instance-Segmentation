import torch
import torchvision.ops as ops

def get_filtered_bboxes(bboxes):
    n = len(bboxes)

    # Create matrix of IOU scores
    iou_matrix = ops.box_iou(bboxes, bboxes)

    # Make matrix upper triangular
    iou_matrix = torch.triu(iou_matrix)

    # Remove columns with max value greater than 0.8
    max_values, _ = torch.max(iou_matrix, dim=0)
    columns_to_keep = max_values <= 0.8

    # Return remaining columns
    filtered_bboxes = [bboxes[i] for i in range(n) if columns_to_keep[i]]
    return filtered_bboxes