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
from natsort import natsorted
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

# models https://github.com/alecpeltekian/mmclassification/blob/15cd34bbef84b103970360c6a35fd6070445c201/docs/model_zoo.md
seed.init_seed()
# LSA KDE mistake https://github.com/ICECCS2020/DRtest/blob/master/coverage_criteria/surprise_coverage.py

# device = torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_model = "cifar100-res18"
# dataset_model = "imagenet-res18"
# dataset_model = "cifar10-res18"
# dataset_model = "stl10-res18"
# dataset_model = "cifar10-cnn"

data_dir = "project_data/"
save_dir = "project_data/experiment1/results_parts/"
# i = 183
i = 1
while os.path.exists(save_dir + "run%s" % i):
    i += 1
    time.sleep(0.01)

# baseline_dir = os.path.dirname(save_path + "{}/{}/{}".format(model_name, name, 'baseline'))
# Creating save folders
if not os.path.exists(save_dir + "run%s" % i):
    os.makedirs(save_dir + "run%s" % i)

init_save_dir = save_dir + "run%s" % i

if dataset_model == "imagenet-res18":
    experiments = experiment_utils.load_1_imagenet(data_dir)
elif dataset_model == "cifar10-res18":
    experiments = experiment_utils.load_1_cifar10_res18(data_dir)
elif dataset_model == "stl10-res18":
    experiments = experiment_utils.load_1_stl10_res18(data_dir)
elif dataset_model == "cifar100-res18":
    experiments = experiment_utils.load_1_cifar100_res18(data_dir)
# elif dataset_model == "cifar10-cnn":
#     experiments = experiment_utils.load_1_cifar10(data_dir)
else:
    exit()

# import cv2
# def saveCifarImage(array, path, file):
#     # array is 3x32x32. cv2 needs 32x32x3
#     array = array.cpu().detach().numpy().transpose(1,2,0)
#     # array is RGB. cv2 needs BGR
#     array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
#     # save to PNG file
#     return cv2.imwrite(path+file+".png", array)

# def check_accuracy(test_loader, model, device):
#     num_correct = 0
#     total = 0
#     model.eval()

#     with torch.no_grad():
#         for data, labels in test_loader:
#             data = data.to(device=device)
#             labels = labels.to(device=device)

#             predictions = model(data)
#             for im in data:
#                 saveCifarImage(im, "project_data/test/", "test")
#             num_correct += (predictions == labels).sum()
#             total += labels.size(0)

#         print(f"Test Accuracy of the model: {float(num_correct)/float(total)*100:.2f}")

for experiment in experiments:
    results = {}
    print(f"Experiment {experiment['name']} started")
    print(f"In folder {init_save_dir}")
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

    dataset, targets = [], []
    dataset_reshaped, targets_reshaped = [], []
    data_loader = iter(data_loader)

    datasets__ = []
    for i in range(dataset_max):
        data, target = next(data_loader)
        data, target = data.to(device), target.to(device)

        dataset.append(data)
        targets.append(target)
        for j in range(len(target)):
            targets_reshaped.append(target[j].cpu().numpy())
            dataset_reshaped.append(data[j].cpu().numpy().reshape(-1))

    # del data_loader
    dataset_permuted = torch.stack(dataset).permute(0, 1, 3, 4, 2)
    dataset = torch.stack(dataset)
    targets = torch.stack(targets)

    save_dir = init_save_dir + "/{}".format(name)
    svm_dir = save_dir
    save_dir = save_dir + "/{}/".format(model_name) #+norm)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("save folder created: ", save_dir)

    # Data coverage
    equivalence_part = coverage_methods.eq_part(targets)
    Centroid_part = coverage_methods.cent_pos(dataset, targets, bound_threshold=12)

    results['dataset'] = name
    results['model_name'] = model_name
    results['equivalence_part'] = equivalence_part
    results['Centroid_part'] = Centroid_part

    if device == torch.device("cuda:0"):
        model = torch.nn.DataParallel(model)
        # cudnn.benchmark = True
        model.cuda()
        model.to(torch.device('cuda'))
        print("Cuda is enabled")
    # check_accuracy(data_loader, model, device)

    criterion = experiment['criterion']
    optimizer = experiment['optimizer'](
        model.parameters(), lr=experiment['lr'])

    epsilons = experiment['epsilons']
    attack = experiment['attack']

    print("Loading foolbox model")

    # Attack
    fmodel = foolbox.PyTorchModel(
        model=model.eval(), bounds=experiment['bounds'], device=device)
    # fmodel = foolbox.PyTorchModel(model.eval(), bounds=(0, 255), device=device)

    bound_part = coverage_methods.bound_cond(fmodel, dataset, targets, bound_threshold=(-40, 40))
    pairw_part = coverage_methods.pairw_cond(fmodel, dataset, targets, bound_threshold=(-30, 30))

    results['bound_part'] = bound_part
    results['pairw_part'] = pairw_part

    # cc = contribution_coverage.output_contributions(fmodel, dataset)
    # Surprise adequacy #
    layer_types = [torch.nn.modules.Module, torch.nn.modules.Conv2d, torch.nn.modules.conv.Conv2d]
    results['lsa_values'] = surprise_adequacy.calculate(model, dataset, layer_types, experiment['nb_classes'], experiment['batch_size'])

    # DNN coverage
    clipped = []
    accuracies = []
    print("Running attacks")
    
    for i in range(len(dataset)):
        raw_advs, clipped_advs, success = attack(
            fmodel, dataset[i], targets[i], epsilons=epsilons)
        clips = []
        for clip in clipped_advs:
            clips.append(clip.cpu())
        clipped.append(clips)
        # clipped.append(clipped_advs)
        del raw_advs, clipped_advs

    print("Calculating FOL")
    fol_data = {"L2": {}, "Linf": {}}
    for key, eps in enumerate(epsilons):
        fol_data['L2'][eps] = fol.L2(fmodel, targets, dataset, clipped, key, device)
        fol_data['Linf'][eps] = fol.Linf(fmodel, targets, dataset, clipped, criterion, key, device)

    results['fol_L2'] = fol_data['L2']
    results['fol_Linf'] = fol_data['Linf']

    # Robustness
    # emp. robustness
    print("Calculating robustness")
    emp_f1, results['emperical_robustness'] = emperical.calculate_f1(fmodel, dataset, clipped, targets, epsilons)
    emp_logit, results['emperical_robustness_logits'] = emperical.calculate_logit(fmodel, dataset, clipped, targets, epsilons)

    # completeness
    print("Calculating AMISE")
    results['amise_f1'] = amise.calculate_f1(emp_f1) #, fmodel, dataset, clipped, targets, epsilons)
    results['amise'] = amise.calculate_logit(emp_logit) #, fmodel, dataset, clipped, targets, epsilons)

    print("Saving")
    save_file = f"{experiment['name']}.pickle"
    result_vars = [results, experiment]

    with open(save_dir + save_file, 'wb') as handle:
        pickle.dump(result_vars, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print("Saving emp values")
    # emp_save_file = f"emp_{experiment['name']}.pickle"
    # emp_result_vars = [emp_f1, emp_logit]

    # with open(save_dir + emp_save_file, 'wb') as handle:
    #     pickle.dump(emp_result_vars, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"{save_dir + save_file} saved")

    gc.collect()
    if device == torch.device("cuda:0"):
        torch.cuda.empty_cache()