import sys
import torch
import torchvision
from torchvision.datasets.coco import CocoDetection
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

sys.path.insert(0, './')

from models.torchvision_bugfix.models.detection.faster_rcnn import FasterRCNN, AnchorGenerator, mobilenet_backbone


def load_with_backbone(pre_trained=True, frozen=True, path=None, device=None, classes=1280):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    backbone = mobilenet_backbone("mobilenet_v3_large", pretrained=False, fpn=True, trainable_layers=3)
    backbone = backbone.eval()
    backbone.training = False

    anchor_sizes = ((32, 64, 128, 256, 512,),) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    num_classes = 91

    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios) 

    path = "models/torchvision_bugfix/detection_checkpoint.model"
    if pre_trained and path is not None:
        model = FasterRCNN(num_classes=num_classes, backbone=backbone,
            rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios))
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        model = FasterRCNN(num_classes=num_classes, backbone=backbone,
            rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios))

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()

    if frozen:
        model = model.eval()
        model.training = False

    return model.__class__.__name__, model
