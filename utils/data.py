import numpy as np
from abc import ABC
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as F


def calculate_distribution(dataset):
    # if hasattr(dataset, 'dataset'):
    #     dataset = dataset.dataset

    class_count = {}
    target_list = []
    for data, target in dataset:
        target_list.append(target)
        key = target
        if key not in class_count:
            class_count[key] = 0

        class_count[key] += 1

    class_count = [*class_count.values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    target_list = torch.tensor(target_list)

    target_list = target_list[torch.randperm(len(target_list))]

    class_weights_all = class_weights[target_list]

    return class_weights_all


def loguniform(a, b, q=None, samples=10, type=float):
    rounding_decimals = 6 if type == float else 0
    return np.round(np.exp(np.random.uniform(a, b, (samples,))), rounding_decimals)


class Transform(ABC):
    """
    Marker class for all transforms
    """
    input_transform = None
    target_transform = None
    params = {}

    def __init__(self, apply_to_input=True, apply_to_target=True, **kwargs):
        self.kwargs = kwargs  # todo remove/refactor
        self._apply_to_input = apply_to_input
        self._apply_to_target = apply_to_target
        self.verify()

    def __call__(self, input, target):
        if self._apply_to_input and self.input_transform is not None and input is not None:
            input = self.input_transform(input)
        if self._apply_to_target and self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return input, target

    def verify(self) -> None:
        """ Method for verifying parameters needed for the Transform class """
        print("not implemented")

    @property
    def name(self):
        return self.__class__.__name__


class COCO_to_RCNN(Transform):
    ignore_mask = True

    def target_transform(self, target):
        image_id = target[0]["image_id"]
        image_id = torch.tensor([image_id])
        anno = [obj for obj in target if obj['iscrowd'] == 0]
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]

        # warning: skip check based on image size (if reshaped)
        w = None
        h = None
        if w is not None and h is not None:
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        segmentations = [obj["segmentation"] for obj in anno]
        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if not self.ignore_mask:
            def convert_coco_poly_to_mask(segmentations, height, width):
                masks = []
                for polygons in segmentations:
                    rles = coco_mask.frPyObjects(polygons, height, width)
                    mask = coco_mask.decode(rles)
                    if len(mask.shape) < 3:
                        mask = mask[..., None]
                    mask = torch.as_tensor(mask, dtype=torch.uint8)
                    mask = mask.any(dim=2)
                    masks.append(mask)
                if masks:
                    masks = torch.stack(masks, dim=0)
                else:
                    masks = torch.zeros((0, height, width), dtype=torch.uint8)
                return masks

            masks = convert_coco_poly_to_mask(segmentations, h, w)
            masks = masks[keep]
        else:
            masks = None
        if keypoints is not None:
            keypoints = keypoints[keep]
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if masks is not None:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd
        return target

class CenterCrop(Transform):
    params = {"size": (int, tuple, list)}
    def __init__(self, apply_to_input=True, apply_to_target=True, **kwargs):
        super().__init__(apply_to_input, apply_to_target, **kwargs)
        self.input_transform = T.CenterCrop(**kwargs)
        self.target_transform = T.CenterCrop(**kwargs)


class ToTensor(Transform):
    input_transform = T.ToTensor()
    def target_transform(self, target):
        target = torch.LongTensor(np.asarray(target))
        if self.kwargs.get('add_channel_to_target', True):
            return target.unsqueeze(0)
        else:
            return target
