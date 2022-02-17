import numpy as np
import sys
import torch
from collections import Counter
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, './')

from utils.metric import Metric, MetricWithSimpleState


def _intersection_over_union_boxes(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def _mAP_coco_evaluator(pred_boxes, targets, dataset):
    raise DeprecationWarning
    outputs = [{k: v.cpu() for k, v in t.items()} for t in pred_boxes]
    res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

    coco = get_coco_api_from_dataset(dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    for v in coco_evaluator.coco_eval.values():
        return v.stats[0]


def _mAP(
        pred_boxes_batch, true_boxes_batch, iou_threshold=0.5, box_format="midpoint", num_classes=91,
        method="continuous"
):
    """
    Calculates mean average precision
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
        method (str): 'interp' or 'continuous' for area under curve calculation
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # pred_boxes = rcnn_2_boxes(pred_boxes)
    # true_boxes = rcnn_2_boxes(true_boxes)

    # used for numerical stability later on
    epsilon = 1e-6

    # list storing all mAP for respective batch idx
    map_batch = []
    for image_idx, (pred_boxes, true_boxes) in enumerate(zip(pred_boxes_batch, true_boxes_batch)):
        # list storing all AP for respective classes
        average_precisions = []
        for c in range(num_classes):
            detections = []
            ground_truths = []

            # Go through all predictions and targets,
            # and only add the ones that belong to the
            # current class c and current batch_idx
            for detection in pred_boxes:
                if detection[1] == c:
                    detections.append(detection)

            for true_box in true_boxes:
                if true_box[1] == c:
                    ground_truths.append(true_box)

            # find the amount of ground truth bboxes for the
            # training example and build empty tensor of length
            amount_bboxes = torch.zeros(len(ground_truths))

            # sort by box probabilities which is index 2
            detections.sort(key=lambda x: x[2], reverse=True)
            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            total_true_bboxes = len(ground_truths)

            # If none exists for this class then we can safely skip
            if total_true_bboxes == 0:
                continue

            for detection_idx, detection in enumerate(detections):
                best_iou = 0

                for idx, gt in enumerate(ground_truths):
                    iou = _intersection_over_union_boxes(
                        torch.tensor(detection[3:]),
                        torch.tensor(gt[3:]),
                        box_format=box_format,
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou > iou_threshold:
                    # only detect ground truth detection once
                    if amount_bboxes[best_gt_idx] == 0:
                        # true positive and add this bounding box to seen
                        TP[detection_idx] = 1
                        amount_bboxes[best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1

                # if IOU is lower then the detection is a false positive
                else:
                    FP[detection_idx] = 1

            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
            precisions = torch.cat((torch.tensor([0]), precisions, torch.tensor([0])))
            recalls = torch.cat((torch.tensor([0]), recalls, torch.tensor([1])))
            # AP from area under curve
            if method == 'interp':
                ap = torch.trapz(precisions, recalls)
            else:
                i = torch.where(recalls[1:] - recalls[:-1])[0]  # points where x axis (recall) changes
                ap = torch.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])  # area under curve
            average_precisions.append(ap)
        mean_ap_image = sum(average_precisions) / (len(average_precisions) + 1e-10)
        map_batch.append(mean_ap_image.item() if not isinstance(mean_ap_image, float) else mean_ap_image)
    return map_batch


class mAP(MetricWithSimpleState):
    target_format_metric = 'boxes'

    @Metric.try_me
    @Metric.adapt_labels
    def __call__(self, *args, **kwargs):
        true_boxes, pred_boxes = args
        iou_threshold = kwargs.get('iou_threshold', self.params.get('iou_threshold', 0.5))
        box_format = kwargs.get('box_format', self.params.get('box_format', 'midpoint'))
        num_classes = kwargs.get('num_classes', self.params.get('num_classes', 91))
        method = kwargs.get('method', self.params.get('method', 'continuous'))
        batch = _mAP(pred_boxes, true_boxes, iou_threshold=iou_threshold, box_format=box_format,
                     num_classes=num_classes,
                     method=method)

        self._set_state(batch)
        return batch

    def __str__(self):
        return f'mAP[IoU={self.params.get("iou_threshold", 0.5):0.2f}]'