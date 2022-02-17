import torch
import itertools

def _calculate_intersection(targets, preds):
    """
    fast calculation of the mean-best-iOU for bounding box detection
    :param targets: the target bounding boxes
    :param preds: the predicted bounding boxes
    :return: the maximum possible iOU value assuming box labels are ideal
    """
    best_scores = {}
    # determine the (x, y)-coordinates of the intersection rectangle
    for boxA, boxB in itertools.product(targets, preds):
        if sum(torch.eq(boxA, boxB)) != 4:
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = interArea / float(boxAArea + boxBArea - interArea)
            if boxB not in best_scores:
                best_scores[boxB] = iou if iou < 1 else 0.0
            else:
                best_scores[boxB] = max(iou, best_scores[boxB]) if iou < 1.0 else best_scores[boxB]
    # return the mean intersection over union value
    return torch.mean(torch.tensor(list(best_scores.values())).unsqueeze(0))