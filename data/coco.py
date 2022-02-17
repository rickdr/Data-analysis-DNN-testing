import sys
import numpy as np
from typing import Any, Callable, cast, Dict, Optional, Tuple, List
import torch
import torchvision
from abc import ABC
from torchvision import transforms as T
from torch.utils.data import Sampler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.functional as F

sys.path.insert(0, './')

from models.torchvision_bugfix.transforms.transforms import Compose 
import utils.data as data_utils 


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
            self,
            root: str,
            annFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            remove_empty=True,
    ):
        super().__init__(root, annFile, transform, target_transform, transforms)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        if remove_empty:
            self.ids = list(self.coco.imgToAnns.keys())
        else:
            self.ids = list(self.coco.imgs.keys())


def collate_coco(batch):
    input, label = [], []
    for b in batch:
        i, l = b
        input.append(i)
        label.append(l)

    input = torch.stack(input, 0, out=None)
    return input, label


def load_detection(loader=True, batch_sampler=None, sampler=None, batch_size=64, path="/data/input/datasets/COCO_2017/"):
    transforms = [
        data_utils.COCO_to_RCNN(apply_to_input=False, apply_to_target=True),
        data_utils.ToTensor(apply_to_input=True, apply_to_target=False),
        # T.Resize(size=(224, 224), apply_to_input=True, apply_to_target=True),
        data_utils.CenterCrop(size=(480, 600), apply_to_input=True, apply_to_target=False),
    ]

    val = CocoDetection(path+'val2017', path+'annotations/instances_val2017.json', transforms=Compose(transforms))

    if loader is False:
        return "cifar10", train, val

    if batch_sampler is not None:
        val_loader = torch.utils.data.DataLoader(val, batch_sampler=batch_sampler, collate_fn=collate_coco)
        # test_loader = torch.utils.data.DataLoader(test, batch_sampler=batch_sampler)

    else:
        val_sampler = None
        if sampler is not None:
            val_sampler = sampler(val, val.targets)

        val_loader = torch.utils.data.DataLoader(
            val, batch_size=batch_size, shuffle=False, sampler=val_sampler, 
            pin_memory=True, num_workers=4, collate_fn=collate_coco)

    return "coco", val_loader