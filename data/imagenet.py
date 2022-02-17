import os
import sys
import numpy as np
import zipfile
import gdown

import torch
import torchvision
from torchvision import transforms

sys.path.insert(0, './')

from utils.data import calculate_distribution


def load_large(loader=True, batch_sampler=None, sampler=None, batch_size=64, path="./project_data/datasets/imagenet"):
    # train_transforms = transforms.Compose([transforms.RandomResizedCrop(225),
    #                                     transforms.CenterCrop(224),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize([0.485, 0.456, 0.406],
    #                                                             [0.229, 0.224, 0.225])])

    # # test_transforms = transforms.Compose([transforms.Resize(255),
    # #                                     transforms.CenterCrop(224),
    # #                                     transforms.ToTensor(),
    # #                                     transforms.Normalize([0.485, 0.456, 0.406],
    # #                                                         [0.229, 0.224, 0.225])])

    # val_transforms = transforms.Compose([transforms.Resize(255),
    #                                     transforms.CenterCrop(224),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize([0.485, 0.456, 0.406],
    #                                                         [0.229, 0.224, 0.225])])

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)]
    )

    VAL_TRANSFORM = transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=imagenet_mean, std=imagenet_std)]
    )

    train = torchvision.datasets.ImageFolder(path + '/train', transform=TRAIN_TRANSFORM)

    val_idcs = np.random.choice(len(train), size=20000, replace=False)
    train_idcs = np.setdiff1d(np.arange(len(train)), val_idcs)

    train_dataset = torch.utils.data.dataset.Subset(
        train, train_idcs)
    val_dataset = torch.utils.data.dataset.Subset(
        train, val_idcs)

    test_dataset = torchvision.datasets.ImageFolder(path+ '/val', transform=VAL_TRANSFORM)

    # test_dataset = torch.utils.data.Subset(test_dataset, np.random.choice(len(test_dataset), 10000 - 1, replace=False))
    # test = torchvision.datasets.ImageFolder(path + '/test', transform=test_transforms)

    # train, test = torch.utils.data.random_split(
    #     train, [int(len(train) * 0.9), int(len(train) * 0.1)])

    if loader is False:
        return "imagenet", train_dataset, val

    if batch_sampler is not None:
        train_batch_sampler = batch_sampler(train_dataset.dataset.targets, batch_size=batch_size)
        val_batch_sampler = batch_sampler(val_dataset.dataset.targets, batch_size=batch_size)
        test_batch_sampler = batch_sampler(test_dataset.targets, batch_size=batch_size)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=batch_sampler)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=batch_sampler)

    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        if sampler is not None:
            # class_weights = calculate_distribution(train_dataset)
            train_sampler = sampler(train_dataset.dataset, train_dataset.dataset.targets)# , weights=class_weights)
            # class_weights = calculate_distribution(val_dataset)
            val_sampler = sampler(val_dataset.dataset, val_dataset.dataset.targets)# , weights=class_weights)
            # class_weights = calculate_distribution(test_dataset)
            test_sampler = sampler(test_dataset, test_dataset.targets)# , weights=class_weights)

        workers = 2
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=0, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=(val_sampler is None),
            pin_memory=True, sampler=val_sampler)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=(test_sampler is None), 
            num_workers=workers, pin_memory=True, sampler=test_sampler)

    return "imagenet", train_loader, val_loader, test_loader


def load(loader=True, sampler=None, batch_size=100, path="./project_data"):
    print("DO YOU WANT TO LOAD THE FULL OR TINY DATASET?")
    return "DO YOU WANT TO LOAD THE FULL OR TINY DATASET?"

    # Imagenet
    # !gdown https://drive.google.com/uc?id=1aScslXi9lg9Fna476hBkHGMMBFuG78zk 
    # IN_ZIP_PATH = 'drive/MyDrive/Navinfo/Project/tiny-imagenet-200.zip'
    # path = os.path.dirname(path)
    path_datasets = path + '/datasets/'

    data_path = os.path.join(path_datasets, 'tiny-imagenet')
    zip_path = os.path.join(path_datasets, 'tiny-imagenet.zip')

    if not os.path.exists(data_path):
        # gdown.download('https://drive.google.com/uc?id=0B9P1L--7Wd2vNm9zMTJWOGxobkU', zip_path, quiet=False)
        if not os.path.exists(zip_path):
            print("download zip")
            return

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path_datasets)

        print("Tiny Imagenet downloaded")
        # tar = tarfile.open(zip_path, 'r')
        # for item in tar:
        #     tar.extract(item, data_path)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train = torchvision.datasets.ImageFolder(os.path.join(data_path, 'train'), transform)
    val = torchvision.datasets.ImageFolder(os.path.join(data_path, 'val'), transform)
    test = torchvision.datasets.ImageFolder(os.path.join(data_path, 'test'), transform)

    if loader is False:
        return "imagenet", train, test, val

    train_shuffle = True
    train_sampler = None
    val_sampler = None
    test_sampler = None
    if sampler is not None:
        train_shuffle = False

        class_weights = calculate_distribution(train)
        train_sampler = sampler(weights=class_weights, num_samples=len(class_weights), replacement=True, shuffle=True)

        class_weights = calculate_distribution(val)
        val_sampler = sampler(weights=class_weights, num_samples=len(class_weights), replacement=True, shuffle=True)

        class_weights = calculate_distribution(test)
        test_sampler = sampler(weights=class_weights, num_samples=len(class_weights), replacement=True, shuffle=True)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=train_shuffle, sampler=train_sampler, pin_memory=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=1)

    return "imagenet_tiny", train_loader, val_loader, test_loader

def load_tiny(loader=True, sampler=None, batch_size=100, path="./project_data"):
    # Imagenet
    # !gdown https://drive.google.com/uc?id=1aScslXi9lg9Fna476hBkHGMMBFuG78zk 
    # IN_ZIP_PATH = 'drive/MyDrive/Navinfo/Project/tiny-imagenet-200.zip'
    # path = os.path.dirname(path)
    path_datasets = path + '/datasets/'

    data_path = os.path.join(path_datasets, 'tiny-imagenet-200')
    zip_path = os.path.join(path_datasets, 'tiny-imagenet-200.zip')

    if not os.path.exists(data_path):
        # gdown.download('https://drive.google.com/uc?id=0B9P1L--7Wd2vNm9zMTJWOGxobkU', zip_path, quiet=False)
        if not os.path.exists(zip_path):
            print("download zip")
            return

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path_datasets)

        print("Tiny Imagenet downloaded")
        # tar = tarfile.open(zip_path, 'r')
        # for item in tar:
        #     tar.extract(item, data_path)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train = torchvision.datasets.ImageFolder(os.path.join(data_path, 'train'), transform)
    val = torchvision.datasets.ImageFolder(os.path.join(data_path, 'val'), transform)
    test = torchvision.datasets.ImageFolder(os.path.join(data_path, 'test'), transform)

    if loader is False:
        return "imagenet_tiny", train, test, val

    train_shuffle = True
    train_sampler = None
    val_sampler = None
    test_sampler = None
    if sampler is not None:
        train_shuffle = False

        class_weights = calculate_distribution(train)
        train_sampler = sampler(weights=class_weights, num_samples=len(class_weights), replacement=True, shuffle=True)

        class_weights = calculate_distribution(val)
        val_sampler = sampler(weights=class_weights, num_samples=len(class_weights), replacement=True, shuffle=True)

        class_weights = calculate_distribution(test)
        test_sampler = sampler(weights=class_weights, num_samples=len(class_weights), replacement=True, shuffle=True)
        print("sampler " + sampler.__class__.__name__ + " is used")

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=train_shuffle, sampler=train_sampler, pin_memory=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=1)

    return "imagenet_tiny", train_loader, val_loader, test_loader

if __name__ == "__main__":
    load_large()
    print("Done")

# def load_tiny():
#     # Imagenet
#     # !gdown https://drive.google.com/uc?id=1aScslXi9lg9Fna476hBkHGMMBFuG78zk 
#     # IN_ZIP_PATH = 'drive/MyDrive/Navinfo/Project/tiny-imagenet-200.zip'
#     path_datasets = os.path.dirname('./data/datasets/')
#     data_path = os.path.join(path_datasets, 'imagenet_tiny')
#     zip_name = 'imagenet_tiny.tgz'

#     if not os.path.exists(data_path):
#         gdown.download('https://drive.google.com/uc?id=0B9P1L--7Wd2vNm9zMTJWOGxobkU', zip_name, quiet=False)

#         with zipfile.ZipFile(os.path.join(path_datasets, zip_name), 'r') as zip_ref:
#             zip_ref.extractall(path_datasets)

#         # tar = tarfile.open(zip_path, 'r')
#         # for item in tar:
#         #     tar.extract(item, data_path)

#     all_classes = {}
#     for i, line in enumerate(open(os.path.join(data_path, 'words.txt'), 'r')):
#         line_split = line.split('\t')
#         if len(line_split) < 2:
#             print(line_split)
#             continue

#         n_id, words = line_split[:2]
#         all_classes[n_id] = words.rstrip()

#     training_data_list = []
#     for folder in os.listdir(os.path.join(data_path, 'train/')):
#         label = folder
#         if folder in all_classes:
#             label = all_classes[folder]

#         for file in os.listdir(os.path.join(data_path, 'train/') + folder + '/images/'):
#             file_dir = os.path.join(data_path, 'train/') + folder + '/images/' + file
#             training_data_list.append((file_dir, label))

#     IN_train_df = pd.DataFrame(training_data_list, columns=['path', 'label'])

#     test_data_list = []
#     for folder in os.listdir(os.path.join(data_path, 'train/')):
#         for file in os.listdir(os.path.join(data_path, 'train/') + folder + '/images/'):
#             file_dir = os.path.join(data_path, 'train/') + folder + '/images/' + file
#             test_data_list.append((file_dir, label))

#     IN_train_df = pd.DataFrame(test_data_list, columns=['path', 'label'])

#     val_data_list = []
#     with open(os.path.join(data_path, 'val/val_annotations.txt'), 'r') as f:
#         for line in f.readlines():
#             file, label = line.split()[0:2]
#             if label in all_classes:
#                 label = all_classes[label]

#         file_dir = os.path.join(data_path, 'val/images/') + file
#         val_data_list.append((file_dir, label))

#     IN_val_df = pd.DataFrame(val_data_list, columns=['path', 'label'])


#     return "imagenet_tiny", IN_train, IN_test, IN_val