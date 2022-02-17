import sys
import torchvision
from torchvision import transforms
import torch

from utils.data import calculate_distribution

def load(loader=True, sampler=None, batch_size=100, path="./project_data"):
    # Mnist
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train = torchvision.datasets.MNIST(root=path + '/datasets', train=True, transform=transform, download=True)
    test = torchvision.datasets.MNIST(root=path + '/datasets', train=False, transform=transform, download=True)

    train, val = torch.utils.data.random_split(train, [int(len(train) * 0.9), int(len(train) * 0.1)])

    if loader is False:
        return "mnnist", train, test, val

    train_shuffle = True
    train_sampler = None
    val_sampler = None
    test_sampler = None
    if sampler is not None:
        train_shuffle = False

        class_weights = calculate_distribution(train)
        train_sampler = sampler(weights=class_weights, num_samples=len(class_weights), replacement=True, shuffle=True
        )

        class_weights = calculate_distribution(val)
        val_sampler = sampler(weights=class_weights, num_samples=len(class_weights), replacement=True, shuffle=True
        )

        class_weights = calculate_distribution(test)
        test_sampler = sampler(weights=class_weights, num_samples=len(class_weights), replacement=True, shuffle=True
        )

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=train_shuffle, sampler=train_sampler, pin_memory=True, num_workers=100)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=1)

    return "mnnist", train_loader, val_loader, test_loader
