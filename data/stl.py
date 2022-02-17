import numpy as np
import torch
import torchvision
from torch.utils.data import Sampler
from torchvision import transforms
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from utils.data import calculate_distribution

def load_10(loader=True, batch_sampler=None, sampler=None, batch_size=64, path="./project_data"):
    # STL10
    # class_mapping = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}
    # class_mapping = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 7, 7: -1, 8: 8, 9: 9}
    class_mapping = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    ttransforms = transforms.Compose([
        lambda x: x
        # lambda x: torch.tensor(class_mapping[x])
    ])

    train = torchvision.datasets.STL10(
        root=path+'/datasets', split="train", transform=transform, target_transform=ttransforms, download=True)
    test = torchvision.datasets.STL10(
        root=path+'/datasets', split="test", transform=transform, target_transform=ttransforms, download=True)

    filter_classmapping_train = [i for i, e in enumerate(train.labels) if e in {key: value for (key, value) in class_mapping.items() if value == -1}]
    filter_classmapping_test = [i for i, e in enumerate(test.labels) if e in {key: value for (key, value) in class_mapping.items() if value == -1}]

    train.data = np.delete(train.data, filter_classmapping_train, axis=0)
    train.labels = np.delete(train.labels, filter_classmapping_train, axis=0)

    test.data = np.delete(test.data, filter_classmapping_test, axis=0)
    test.labels = np.delete(test.labels, filter_classmapping_test, axis=0)

    if loader is False:
        return "STL10", train, test

    train, val = torch.utils.data.random_split(
        train, [int(len(train) * 0.9), int(len(train) * 0.1)])

    # train_indices, val_indices = train_test_split(list(range(len(train.targets))), test_size=0.1, stratify=train.targets)
    # train = torch.utils.data.Subset(train, train_indices)
    # val = torch.utils.data.Subset(train, val_indices)

    if batch_sampler is not None:
        train_batch_sampler = batch_sampler(train.dataset.targets, batch_size=batch_size)
        val_batch_sampler = batch_sampler(val.dataset.targets, batch_size=batch_size)
        test_batch_sampler = batch_sampler(test.targets, batch_size=batch_size)

        train_loader = torch.utils.data.DataLoader(train, batch_sampler=batch_sampler)
        val_loader = torch.utils.data.DataLoader(val, batch_sampler=batch_sampler)
        test_loader = torch.utils.data.DataLoader(test, batch_sampler=batch_sampler)
    else:
        train_shuffle = True
        train_sampler = None
        val_sampler = None
        test_sampler = None
        if sampler is not None:
            train_shuffle = False
            class_weights = calculate_distribution(train)
            train_sampler = sampler(train.dataset, train.dataset.targets)# , weights=class_weights)
            class_weights = calculate_distribution(val)
            val_sampler = sampler(val.dataset, val.dataset.targets)# , weights=class_weights)
            class_weights = calculate_distribution(test)
            test_sampler = sampler(test, test.targets)# , weights=class_weights)

        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=train_shuffle, sampler=train_sampler, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val, batch_size=batch_size, shuffle=False, sampler=val_sampler)
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=batch_size, shuffle=False, sampler=test_sampler)

    return "STL10", train_loader, val_loader, test_loader