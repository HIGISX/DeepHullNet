import torch
from typing import Union
from scipy.spatial import ConvexHull
from shapely import geometry

TOKENS = {
    '<sos>': 0,
    '<eos>': 1
}
class AverageMeter(object):
    """
    Computes and stores the average and current value

    Adapted from pointer-networks-pytorch by ast0414:
      https://github.com/ast0414/pointer-networks-pytorch
    """

    def __init__(self):
        self.history = []
        self.reset(record=False)

    def reset(
            self,
            record: bool = True
    ):
        if record:
            self.history.append(self.avg)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(
            self,
            val: Union[float, int],
            n: int = 1
    ):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def masked_accuracy(
        output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
) -> float:
    """
    Compute accuracy of softmax output with mask applied over values.

    Adapted from pointer-networks-pytorch by ast0414:
      https://github.com/ast0414/pointer-networks-pytorch
    """

    with torch.no_grad():
        masked_output = torch.masked_select(output, mask)
        masked_target = torch.masked_select(target, mask)
        accuracy = masked_output.eq(masked_target).float().mean()
        seq_acc = torch.tensor(0)
        if accuracy == 1:
            seq_acc = accuracy
        return accuracy, seq_acc

def calculate_hull_overlap(data, length, pointer_argmaxs):
    """
    Compute the percent overlap between the predicted and true convex hulls.
    """

    points = data[2:length, :2]
    pred_hull_idxs = pointer_argmaxs[pointer_argmaxs > 1] - 2
    true_hull_idxs = ConvexHull(points).vertices.tolist()
    if len(pred_hull_idxs) >= 3 and len(true_hull_idxs) >= 3:
        shape_pred = geometry.Polygon(points[pred_hull_idxs])
        shape_true = geometry.Polygon(points[true_hull_idxs])
        if shape_pred.is_valid and shape_true.is_valid:
            area = shape_true.intersection(shape_pred).area
            percent_area = area / max(shape_pred.area, shape_true.area)
        else:
            percent_area = 0.0
    else:
        percent_area = 0.0
    return percent_area