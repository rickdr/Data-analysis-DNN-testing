import sys
import numpy as np
import torch
from torch import nn

sys.path.insert(0, './')

from models import CNN, resnet, vgg
from data import mnist, cifar, stl, imagenet, sampling
from project.attacks.attacks import L2FastGradientAttack, LinfFastGradientAttack, L2ProjectedGradientDescentAttack, LinfProjectedGradientDescentAttack

cifar10_bs = 10
stl10_bs = 10
cifar100_bs = 5
imagenet_bs = 5

# models/results/basic_CNN/cifar10/run1/
# models/results/four_layered_CNN/cifar10/run1
# project_data/models/results/three_layered_CNN/cifar10/run1

def load_1_cifar10(data_dir):
    number_epsilons = 20
    linfepsilons = np.geomspace(0.00001, 1., number_epsilons)
    l2epsilons = np.geomspace(0.01, 1000, number_epsilons)
    chunks = [10000]
    samps = [None, sampling.BalancedBatchSampler]
    exps = []
    run = 1
    #     chunkz = chunks_none
    # else:
    #     chunkz = chunks
    chunkz = chunks
    for j, chunk in enumerate(chunkz):
        # if samp == None:
        #     continue
        for i, samp in enumerate(samps):
            exps.append(dict(
                name=f"{i}.{chunk}.2",
                chunk=chunk,  # None,
                batch_size=cifar10_bs,
                dataset=cifar.load_10,
                sampler=[i, samp],
                load_model=CNN.load_basic,
                model_path=data_dir + 'models/results/basic_CNN/cifar10/run1/epoch-85.pt',
                # attack=L2FastGradientAttack(),
                attack=L2ProjectedGradientDescentAttack(),
                criterion=nn.CrossEntropyLoss(),
                epsilons=l2epsilons,
                optimizer=torch.optim.Adam,
                lr=0.001,
                norm=2,
                input_shape=(3, 32, 32),
                bounds=(0.0, 1.0),
                nb_classes=10
            ))
            exps.append(dict(
                name=f"{i}.{chunk}.inf",
                chunk=chunk,  # None,
                batch_size=cifar10_bs,
                dataset=cifar.load_10,
                sampler=[i, samp],
                load_model=CNN.load_basic,
                model_path=data_dir + 'models/results/basic_CNN/cifar10/run1/epoch-85.pt',
                # attack=LinfFastGradientAttack(),
                attack=LinfProjectedGradientDescentAttack(),
                criterion=nn.CrossEntropyLoss(),
                epsilons=linfepsilons,
                optimizer=torch.optim.Adam,
                lr=0.001,
                norm=np.inf,
                input_shape=(3, 32, 32),
                bounds=(0.0, 1.0),
                nb_classes=10
            ))

    return exps


def load_1_cifar10_res18(data_dir):
    number_epsilons = 20
    linfepsilons = np.geomspace(0.00001, 1., number_epsilons)
    l2epsilons = np.geomspace(0.01, 1000, number_epsilons)
    # chunks = [10000]
    # chunks = [7000, 10000]
    # chunks = [100]
    # chunks = [5000, 10000]
    chunks = [100, 200, 300, 500, 700, 1000, 1200, 1500, 2000, 3000, 5000, 7000, 10000]
    # chunks = [3000, 5000, 7000, 10000]
    # chunks_none = [10000]
    samps = [None, sampling.BalancedBatchSampler]
    exps = []
    chunkz = chunks
    for chunk in chunkz:
        # if samp == None:
        #     continue
        for i, samp in enumerate(samps):
            exps.append(dict(
                name=f"{i}.{chunk}.2",
                chunk=chunk,  # None,
                batch_size=cifar10_bs,
                dataset=cifar.load_10,
                sampler=[i, samp],
                load_model=vgg.load_16,
                model_path=data_dir + 'models/results/ResNet18/cifar10/run14/epoch-48.pt',
                attack=L2ProjectedGradientDescentAttack(),
                # attack=L2FastGradientAttack(),
                criterion=nn.CrossEntropyLoss(),
                epsilons=l2epsilons,
                optimizer=torch.optim.Adam,
                lr=0.001,
                norm=2,
                input_shape=(3, 32, 32),
                bounds=(0.0, 1.0),
                nb_classes=10
            ))
            exps.append(dict(
                name=f"{i}.{chunk}.inf",
                chunk=chunk,  # None,
                batch_size=cifar10_bs,
                dataset=cifar.load_10,
                sampler=[i, samp],
                load_model=resnet.load_18,
                model_path=data_dir + 'models/results/ResNet18/cifar10/run14/epoch-48.pt', # run 158
                attack=LinfProjectedGradientDescentAttack(),
                # attack=LinfFastGradientAttack(),
                criterion=nn.CrossEntropyLoss(),
                epsilons=linfepsilons,
                optimizer=torch.optim.Adam,
                lr=0.001,
                norm=np.inf,
                input_shape=(3, 32, 32),
                bounds=(0.0, 1.0),
                nb_classes=10
            ))

    return exps

def load_1_stl10_res18(data_dir):
    number_epsilons = 20
    linfepsilons = np.geomspace(0.00001, 1., number_epsilons)
    l2epsilons = np.geomspace(0.01, 1000, number_epsilons)
    # chunks = [10000]
    # chunks = [7000, 10000]
    # chunks = [100, 200, 300, 500, 600, 700, 1000, 1200, 1500, 2000]
    chunks = [100, 200, 300, 500, 700, 1000, 1200, 1500, 2000, 3000, 5000, 7000, 10000]
    # chunks = [10000]
    # chunks_none = [10000]
    samps = [None, sampling.BalancedBatchSampler]
    exps = []
    chunkz = chunks
    for chunk in chunkz:
        # if samp == None:
        #     continue
        for i, samp in enumerate(samps):
            exps.append(dict(
                name=f"{i}.{chunk}.2",
                chunk=chunk,  # None,
                batch_size=stl10_bs,
                dataset=stl.load_10,
                sampler=[i, samp],
                load_model=resnet.load_18,
                model_path=data_dir + 'models/results/ResNet18/cifar10/run14/epoch-48.pt',
                # attack=L2FastGradientAttack(),
                attack=L2ProjectedGradientDescentAttack(),
                criterion=nn.CrossEntropyLoss(),
                epsilons=l2epsilons,
                optimizer=torch.optim.Adam,
                lr=0.001,
                norm=2,
                input_shape=(3, 32, 32),
                bounds=(0.0, 1.0),
                nb_classes=10
            ))
            exps.append(dict(
                name=f"{i}.{chunk}.inf",
                chunk=chunk,  # None,
                batch_size=stl10_bs,
                dataset=stl.load_10,
                sampler=[i, samp],
                load_model=resnet.load_18,
                model_path=data_dir + 'models/results/ResNet18/cifar10/run14/epoch-48.pt', # run 158
                # attack=LinfFastGradientAttack(),
                attack=LinfProjectedGradientDescentAttack(),
                criterion=nn.CrossEntropyLoss(),
                epsilons=linfepsilons,
                optimizer=torch.optim.Adam,
                lr=0.001,
                norm=np.inf,
                input_shape=(3, 32, 32),
                bounds=(0.0, 1.0),
                nb_classes=10
            ))

    return exps

def load_1_cifar100_res18(data_dir):
    number_epsilons = 20
    linfepsilons = np.geomspace(0.00001, 1., number_epsilons)
    l2epsilons = np.geomspace(0.01, 1000, number_epsilons)
    # chunks = [10000]
    chunks = [100, 200, 300]
    # chunks = [100, 200, 300, 500, 700, 1000, 1200, 1500, 2000, 3000, 5000, 7000, 10000]
    # chunks = [1200]
    # chunks = [2000, 3000, 5000, 10000]

    # chunks = [10000]
    # chunks_none = [10000]
    samps = [None, sampling.BalancedBatchSampler]
    exps = []
    chunkz = chunks
    for chunk in chunkz:
        for i, samp in enumerate(samps):
            # if chunk == 700 and samp == None:
            #     continue
            # if samp == None:
            #     continue
            exps.append(dict(
                name=f"{i}.{chunk}.2",
                chunk=chunk,  # None,
                batch_size=cifar100_bs,
                dataset=cifar.load_100,
                sampler=[i, samp],
                load_model=resnet.load_18,
                model_path=data_dir + 'models/results/ResNet18/cifar100/run9/epoch-48.pt',
                attack=L2FastGradientAttack(),
                # attack=L2ProjectedGradientDescentAttack(),
                criterion=nn.CrossEntropyLoss(),
                epsilons=l2epsilons,
                optimizer=torch.optim.Adam,
                lr=0.001,
                norm=2,
                input_shape=(3, 32, 32),
                bounds=(0.0, 1.0),
                nb_classes=100
            ))
            exps.append(dict(
                name=f"{i}.{chunk}.inf",
                chunk=chunk,  # None,
                batch_size=cifar100_bs,
                dataset=cifar.load_100,
                sampler=[i, samp],
                load_model=resnet.load_18,
                model_path=data_dir + 'models/results/ResNet18/cifar100/run9/epoch-48.pt', # run 158
                attack=LinfFastGradientAttack(),
                # attack=LinfProjectedGradientDescentAttack(),
                criterion=nn.CrossEntropyLoss(),
                epsilons=linfepsilons,
                optimizer=torch.optim.Adam,
                lr=0.001,
                norm=np.inf,
                input_shape=(3, 32, 32),
                bounds=(0.0, 1.0),
                nb_classes=100
            ))

    return exps


def load_1_imagenet(data_dir):
    number_epsilons = 20
    linfepsilons = np.geomspace(0.00001, 1., number_epsilons)
    l2epsilons = np.geomspace(0.01, 1000, number_epsilons)
    chunks = [600]
    # chunks = [2000, 3000, 5000, 7000, 10000]
    # chunks = [1200, 1500, 2000]
    # chunks = [500, 700, 1000, 1200, 1500, 2000, 3000, 5000, 7000, 10000]
    samps = [None, sampling.BalancedBatchSampler]
    exps = []
    chunkz = chunks
    for chunk in chunkz:
        for i, samp in enumerate(samps):
            if samp == None:
                continue
            # if chunk == 2000 and samp != None:
            #     continue
            # if chunk != 2000:
            # exps.append(dict(
            #     name=f"{i}.{chunk}.2",
            #     chunk=chunk,  # None,
            #     batch_size=imagenet_bs,
            #     dataset=imagenet.load_large,
            #     # sampler=sampling.BalancedSampler,
            #     sampler=[i, samp],
            #     load_model=resnet.load_18,
            #     model_path=None,
            #     # model_path=data_dir + 'models/results/ResNet18/imagenet/imagenet-zoo.pth',
            #     # attack=L2FastGradientAttack(),
            #     attack=L2ProjectedGradientDescentAttack(),
            #     criterion=nn.CrossEntropyLoss(),
            #     epsilons=l2epsilons,
            #     optimizer=torch.optim.Adam,
            #     lr=0.001,
            #     norm=2,
            #     input_shape=(3, 224, 224),
            #     bounds=(0.0, 1.0),
            #     nb_classes=1000
            # ))

            exps.append(dict(
                name=f"{i}.{chunk}.inf",
                chunk=chunk,  # None,
                batch_size=imagenet_bs,
                dataset=imagenet.load_large,
                # sampler=sampling.BalancedSampler,
                sampler=[i, samp],
                load_model=resnet.load_18,
                model_path=None,
                # model_path=data_dir + 'models/results/ResNet18/imagenet/imagenet-zoo.pth',
                attack=LinfFastGradientAttack(),
                # attack=LinfProjectedGradientDescentAttack(),
                criterion=nn.CrossEntropyLoss(),
                epsilons=linfepsilons,
                optimizer=torch.optim.Adam,
                lr=0.001,
                norm=np.inf,
                input_shape=(3, 224, 224),
                bounds=(0.0, 1.0),
                nb_classes=1000
            ))

    return exps

def load_1_imagenet_resnet50(data_dir):
    number_epsilons = 20
    linfepsilons = np.geomspace(0.00001, 1., number_epsilons)
    l2epsilons = np.geomspace(0.01, 1000, number_epsilons)
    chunks = [1500, 2000, 3000, 5000, 7000, 10000]
    samps = [None, sampling.BalancedBatchSampler]
    exps = []
    for chunk in chunkz:
        # if samp == None:
        #     continue
        for i, samp in enumerate(samps):
            exps.append(dict(
                name=f"{i}.{chunk}.2",
                chunk=chunk,  # None,
                batch_size=imagenet_bs,
                dataset=imagenet.load_large,
                # sampler=sampling.BalancedSampler,
                sampler=[i, samp],
                load_model=resnet.load_50,
                model_path=data_dir + 'models/results/ResNet50/imagenet/imagenet-zoo.pth',
                attack=L2FastGradientAttack(),
                criterion=nn.CrossEntropyLoss(),
                epsilons=l2epsilons,
                optimizer=torch.optim.Adam,
                lr=0.001,
                norm=2,
                input_shape=(3, 224, 224),
                bounds=(0.0, 1.0),
                nb_classes=1000
            ))

            exps.append(dict(
                name=f"{i}.{chunk}.inf",
                chunk=chunk,  # None,
                batch_size=imagenet_bs,
                dataset=imagenet.load_large,
                # sampler=sampling.BalancedSampler,
                sampler=[i, samp],
                load_model=resnet.load_50,
                model_path=data_dir + 'models/results/ResNet50/imagenet/imagenet-zoo.pth',
                attack=LinfFastGradientAttack(),
                criterion=nn.CrossEntropyLoss(),
                epsilons=linfepsilons,
                optimizer=torch.optim.Adam,
                lr=0.001,
                norm=np.inf,
                input_shape=(3, 224, 224),
                bounds=(0.0, 1.0),
                nb_classes=1000
            ))

    return exps