import torch
import torchvision.ops as ops

def fast_nms(bboxes, threshold=0.65):
    """
    Returns a boolean mask of bboxes to keep.
    """
    n = len(bboxes)

    # Create matrix of IOU scores
    iou_matrix = ops.box_iou(bboxes, bboxes)

    # Make matrix upper triangular
    iou_matrix = torch.triu(iou_matrix, diagonal=1)

    # Get columns with max value lower than threshold
    max_values, _ = torch.max(iou_matrix, dim=-2)
    columns_to_keep = torch.where(max_values < threshold)[0]

    return columns_to_keep

def batched_fnms(bboxes, cls, coefficients, threshold=0.75):
    """
    Returns bboxes, cls and coefficients with the same shape as the input, but with the columns
    that have been filtered out masked to 0.
    """
    masked_columns = torch.zeros(len(bboxes), len(bboxes[0]))
    filtered_bboxes = torch.zeros_like(bboxes)
    filtered_cls = torch.zeros_like(cls)
    filtered_coefficients = torch.zeros_like(coefficients)

    for i in range(len(bboxes)):
        columns_to_keep = fast_nms(bboxes[i], threshold=threshold)
        filtered_bboxes[i, columns_to_keep] = bboxes[i, columns_to_keep]
        filtered_cls[i, columns_to_keep] = cls[i, columns_to_keep]
        filtered_coefficients[i, columns_to_keep] = coefficients[i, columns_to_keep]
        masked_columns[i, columns_to_keep] = 1

    return filtered_bboxes, filtered_cls, filtered_coefficients, masked_columns


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

    cls = torch.tensor([[
        [0.98],
        [0.97],
        [0.96],
        [0.95],
        [0.94],
        [0.93],
        [0.92],
        [0.91],
        [0.90],
    ], [   
        [0.98],
        [0.97],
        [0.96],
        [0.95],
        [0.94],
        [0.93],
        [0.92],
        [0.91],
        [0.90],
    ]], dtype=torch.float32)

    coefficients = torch.tensor([[
        [1.0],
        [-1.0],
        [1.0],
        [-1.0],
        [1.0],
        [-1.0],
        [1.0],
        [-1.0],
        [1.0],
    ], [
        [-1.0],
        [1.0],
        [-1.0],
        [1.0],
        [-1.0],
        [1.0],
        [-1.0],
        [1.0],
        [-1.0],
    ]], dtype=torch.float32)


    expected_bboxes = torch.tensor([[
        [0., 0., 10., 10.],
        [1., 1., 12., 12.],
        [2., 2., 14., 14.],
        [3., 3., 16., 16.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [10., 10., 30., 30.],
    ], [   
        [1., 1., 11., 11.],
        [2., 2., 13., 13.],
        [3., 3., 15., 15.],
        [4., 4., 17., 17.],
        [0., 0., 0., 0.],
        [13., 13., 23., 23.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [19., 19., 29., 29.],
    ]])

    expected_cls = torch.tensor([[
        [0.98],
        [0.97],
        [0.96],
        [0.95],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.90],
    ], [
        [0.98],
        [0.97],
        [0.96],
        [0.95],
        [0.0],
        [0.93],
        [0.0],
        [0.0],
        [0.90],
    ]])

    expected_coefficients = torch.tensor([[
        [1.0],
        [-1.0],
        [1.0],
        [-1.0], 
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [1.0],
    ], [
        [-1.0],
        [1.0],
        [-1.0],
        [1.0],
        [0.0],
        [1.0],
        [0.0],
        [0.0],
        [-1.0],
    ]])

    expected_masked_columns = torch.tensor([
        [1., 1., 1., 1., 0., 0., 0., 0., 1.],
        [1., 1., 1., 1., 0., 1., 0., 0., 1.]
    ])


    filtered_bboxes, filtered_cls, filtered_coefficients, masked_columns = batched_fnms(bboxes, cls, coefficients, threshold=0.65)

    assert torch.all(torch.eq(filtered_bboxes, expected_bboxes))
    assert torch.all(torch.eq(filtered_cls, expected_cls))
    assert torch.all(torch.eq(filtered_coefficients, expected_coefficients))
    assert torch.all(torch.eq(masked_columns, expected_masked_columns))

    print("batched_fast_nms test passed!")



def run_tests():
    test_batched_fast_nms()