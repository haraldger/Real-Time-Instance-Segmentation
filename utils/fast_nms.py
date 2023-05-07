import torch
import torchvision.ops as ops

def fast_nms(bboxes, threshold=0.75):
    n = len(bboxes)

    # Create matrix of IOU scores
    iou_matrix = ops.box_iou(bboxes, bboxes)

    # Make matrix upper triangular
    iou_matrix = torch.triu(iou_matrix, diagonal=1)

    # Remove columns with max value greater than threshold
    max_values, _ = torch.max(iou_matrix, dim=-2)
    columns_to_keep = max_values <= threshold

    # Return remaining columns
    filtered_bboxes = bboxes[columns_to_keep]
    return filtered_bboxes

def batched_fnms(bboxes, threshold=0.75):
    bboxes = [ fast_nms(bboxes[i], threshold).unsqueeze(0) for i in range(len(bboxes)) ]
    return bboxes

# Tests
def test_batched_fast_nms():
    print("Running batched_fast_nms test...")

    bboxes = torch.tensor([[
        [0, 0, 10, 10],
        [1, 1, 12, 12],
        [2, 2, 14, 14],
        [3, 3, 16, 16],
        [4, 4, 18, 18],
        [5, 5, 20, 20],
        [6, 6, 22, 22],
        [7, 7, 24, 24],
        [10, 10, 30, 30],
    ], [ 
        [1, 1, 11, 11],
        [2, 2, 13, 13],
        [3, 3, 15, 15],
        [4, 4, 17, 17],
        [5, 5, 19, 19],
        [13, 13, 23, 23],
        [14, 14, 24, 24],
        [15, 15, 25, 25],
        [19, 19, 29, 29],
    ]], dtype=torch.float32)

    filtered_bboxes = batched_fnms(bboxes, threshold=0.65)
    target_bboxes = torch.tensor([[
        [0, 0, 10, 10],
        [1, 1, 12, 12],
        [2, 2, 14, 14],
        [3, 3, 16, 16],
        [10, 10, 30, 30],
    ], [   
        [1, 1, 11, 11],
        [2, 2, 13, 13],
        [3, 3, 15, 15],
        [4, 4, 17, 17],
        [13, 13, 23, 23],
        [19, 19, 29, 29],
    ]], dtype=torch.float32)
    assert torch.allclose(filtered_bboxes, target_bboxes)

    print("batched_fast_nms test passed")

def run_tests():
    test_batched_fast_nms()