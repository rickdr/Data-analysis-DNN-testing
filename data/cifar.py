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
    # CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train = torchvision.datasets.CIFAR10(
        root=path+'/datasets', train=True, transform=transform, download=True)
    test = torchvision.datasets.CIFAR10(
        root=path+'/datasets', train=False, transform=transform, download=True)

    if loader is False:
        return "cifar10", train, test
        

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

    return "cifar10", train_loader, val_loader, test_loader


def load_100(loader=True, batch_sampler=None, sampler=None, batch_size=64, path="./project_data"):
    # CIFAR-100
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train = torchvision.datasets.CIFAR100(
        root=path+'/datasets', train=True, transform=transform, download=True)
    test = torchvision.datasets.CIFAR100(
        root=path+'/datasets', train=False, transform=transform, download=True)

    if loader is False:
        return "cifar100", train, test
        

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

    return "cifar100", train_loader, val_loader, test_loader