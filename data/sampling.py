import sys
import random
import numpy as np
from typing import Optional, Sequence


import torch
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler as _TorchWeightedRandomSampler


class WeightedRandomSampler(_TorchWeightedRandomSampler):
    """
    Enhance PyTorch DistributedSampler to support non-evenly divisible sampling.
    Args:
        dataset: Dataset used for sampling.
        even_divisible: if False, different ranks can have different data length.
            for example, input data: [1, 2, 3, 4, 5], rank 0: [1, 3, 5], rank 1: [2, 4].
        num_replicas: number of processes participating in distributed training.
            by default, `world_size` is retrieved from the current distributed group.
        rank: rank of the current process within `num_replicas`. by default,
            `rank` is retrieved from the current distributed group.
        shuffle: if `True`, sampler will shuffle the indices, default to True.
        kwargs: additional arguments for `DistributedSampler` super class, can be `seed` and `drop_last`.
    More information about DistributedSampler, please check:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler.
    """

    def __init__(
        self,
        weights: Optional[float] = None,
        num_samples: Optional[int] = None,
        replacement: bool = False,
        generator=None,
        **kwargs,
    ):
        super().__init__(weights, num_samples, replacement=replacement, generator=generator)


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None,):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, dataset, idx, labels = None):
        if self.labels is not None:
            return self.labels[idx]
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx]
            elif is_torchvision_installed and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        # return len(self.labels)
        return self.balanced_max*len(self.keys)


# class BalancedSampler(torch.utils.data.sampler.Sampler):    
#     def __init__(self, cost, batch_size):
#         # random.shuffle(cost)
#         index = np.argsort(cost).tolist()
#         chunk_size = int(float(len(cost))/batch_size)
#         self.index = []
#         for i in range(batch_size):
#             self.index.append(index[i*chunk_size:(i + 1)*chunk_size])

#     def _g(self):
#         # shuffle data
#         for index_i in self.index:
#             random.shuffle(index_i)

#         for batch_index in zip(*self.index):
#             yield batch_index

#     def __iter__(self):
#         return self._g()

#     def __len__(self):
#         return len(self.index[0])


class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1)
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0, int(1e8), size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)