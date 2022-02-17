import os
import time
import sys
import copy
import numpy as np
import torch
import foolbox
import random
import gc
import csv
import pickle
from pathlib import Path
from natsort import natsorted
from glob import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib
from tqdm import tqdm
from numba import cuda
import matplotlib.pyplot as plt
from multiprocessing import Process

from art.estimators.classification import PyTorchClassifier

sys.path.insert(0, './')

from utils import seed, experiments as experiment_utils
from project.dnn_coverage import contribution_coverage, surprise_adequacy, fol
from project.robustness import emperical, clever

from project.data_coverage import coverage_methods
from project.completeness import amise

seed.init_seed()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

attack = "PGD"
# dataset_model = "imagenet-res18"
# dataset_model = "cifar10-res18"
dataset_model = "cifar100-res18"

data_dir = "project_data/"
save_dir = "project_data/experiment1/results_parts/"

if dataset_model == "imagenet-res18" and attack == "FGSM":
    i = 2
elif dataset_model == "cifar10-res18" and attack == "FGSM":
    i = 0
elif dataset_model == "cifar100-res18" and attack == "FGSM":
    i = 1
elif dataset_model == "imagenet-res18" and attack == "PGD":
    i = 9
elif dataset_model == "cifar10-res18" and attack == "PGD":
    i = 6
elif dataset_model == "cifar100-res18" and attack == "PGD":
    i = 8
# elif dataset_model == "cifar10-cnn":
#     experiments = experiment_utils.load_1_cifar10(data_dir)
else:
    exit()

# i = 12
if not os.path.exists(save_dir + "run%s" % i):
    os.makedirs(save_dir + "run%s" % i)

init_save_dir = save_dir + "run%s" % i
experiments = []
all_results = []
files = []
for f_name in natsorted(glob(f"{init_save_dir}/**/*.pickle", recursive=True)):
    # version, key, norm = Path(f_name).stem.split(".")
    with open(f_name, 'rb') as handle:
        if "_finished" in f_name or "emp_" in f_name:
            continue
        # key = Path(f_name).stem
        file_data, experiment_settings = pickle.load(handle)

        experiments.append(experiment_settings)
        all_results.append(file_data)
        files.append(f_name)

for k, experiment in enumerate(experiments):
    experiment['batch_size'] = 1
    results = all_results[k]
    print(f"Experiment {experiment['name']} loaded")
    print(f"From folder {init_save_dir}")

    print("Loading model")
    # model loading
    model_name, model = experiment['load_model'](pre_trained=True, frozen=True, device=device, path=experiment['model_path'], classes=experiment['nb_classes'])

    # Loading data
    i_sampler, sampler = experiment['sampler']

    if "imagenet" in dataset_model:
        name, train_loader, val_loader, test_loader = experiment['dataset'](batch_size=experiment['batch_size'], sampler=sampler) #, path=data_dir)
        data_loader = copy.deepcopy(test_loader)
        del train_loader, val_loader, test_loader
    else:
        name, train_loader, val_loader, test_loader = experiment['dataset'](batch_size=experiment['batch_size'], sampler=sampler, path=data_dir)
        data_loader = copy.deepcopy(test_loader)
        del train_loader, val_loader, test_loader

    # Take piece from dataset
    if experiment['chunk'] is None:
        dataset_max = len(data_loader)
    else:
        dataset_max = int(experiment['chunk'] / experiment['batch_size'])
        if dataset_max > len(data_loader):
            raise Exception("Chunk %s is to large", experiment['chunk'])
            break

    dataset = []
    data_loader = iter(data_loader)
    for i in range(dataset_max):
        data, _ = next(data_loader)
        data = data.to(device)
        dataset.append(data)

    del data_loader
    dataset = torch.stack(dataset)

    save_dir = init_save_dir + "/{}".format(name)
    svm_dir = save_dir
    save_dir = save_dir + "/{}/".format(model_name)

    save_file = f"{experiment['name']}_finished.pickle"
    if os.path.isfile(save_dir + save_file):
        continue

    if device == torch.device("cuda:0"):
        model = torch.nn.DataParallel(model)
        # cudnn.benchmark = True
        model.cuda()
        model.to(torch.device('cuda'))
        print("Cuda is enabled")

    criterion = experiment['criterion']
    optimizer = experiment['optimizer'](
        model.parameters(), lr=experiment['lr'])

    epsilons = experiment['epsilons']

    # CLEVER
    norm = experiment['norm']
    nb_classes = experiment['nb_classes']
    inp_shape = experiment['input_shape']

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=inp_shape, # (3, 224, 224),
        channels_first=False,
        nb_classes=nb_classes,
        device_type=device
    )

    print("Calculating CLEVER")
    results['clever'] = clever.untargeted(dataset, classifier, norm, nb_classes, epsilons)

    # emp_save_file = f"emp_{experiment['name']}.pickle"
    # if os.path.isfile(save_dir + emp_save_file):
    #     print("Calculating AMISE")
    #     with open(save_dir + emp_save_file, 'rb') as handle:
    #         emp_f1, emp_logit = pickle.load(handle)

    #     # completeness
    #     results['amise_f1'] = amise.calculate_f1(emp_f1) #, fmodel, dataset, clipped, targets, epsilons)
    #     results['amise'] = amise.calculate_logit(emp_logit) #, fmodel, dataset, clipped, targets, epsilons)

    print("Saving")
    result_vars = [results, experiment]
    with open(save_dir + save_file, 'wb') as handle:
        pickle.dump(result_vars, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"{save_dir}{save_file} saved")
    gc.collect()
    if device == torch.device("cuda:0"):
        torch.cuda.empty_cache()

# os.mkdir(save_dir + "old/")
# for f_name in files:
#     os.rename(save_dir + f_name, save_dir + "old/" + f_name)