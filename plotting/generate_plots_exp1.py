import sys
import os
import time
import copy
import pickle
from pathlib import Path
from natsort import natsorted
from glob import glob
from tqdm import tqdm
from foolbox.models import PyTorchModel
from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import cupy as cp
import numpy as np

import random
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
from matplotlib.gridspec import GridSpec
import gc
import resource
from art.estimators.classification import PyTorchClassifier
from sklearn import metrics
from scipy import stats
from skimage.metrics import structural_similarity
from image_similarity_measures import quality_metrics
import albumentations as albument
from sklearn.metrics import classification_report

sys.path.insert(0, './')

from utils import seed

from data import cifar
# from data.data_sampling import WeightedRandomSampler
# from models import resnet
# from models import vgg
# from models import CNN
# from project import utils
# from project.attacks import attacks as import_attacks
# from project.completeness.wasserstein import wasserstein_distance
# from project.robustness.clever import clever_u
# from project.attacks import noises

# from utils import seed

# %matplotlib widget
# %matplotlib inline

seed.init_seed()

all_colors = [k for k,v in pltc.cnames.items()]
plt.rcParams.update({'figure.max_open_warning': 0})

# attack = "PGD"
# attack = "FGSM"
# dataset = "imagenet"
# dataset = "cifar10"
# dataset = "cifar100"
for dataset in ["cifar10", "cifar100", "imagenet"]:
    for attack in ["FGSM", "PGD"]:
        data_dir = "project_data/"
        save_dir = "project_data/experiment1/results_parts/"

        if dataset == "imagenet" and attack == "FGSM":
            run = 2
        elif dataset == "cifar10" and attack == "FGSM":
            run = 0
        elif dataset == "cifar100" and attack == "FGSM":
            run = 1
        elif dataset == "imagenet" and attack == "PGD":
            run = 9
        elif dataset == "cifar10" and attack == "PGD":
            run = 6
        elif dataset == "cifar100" and attack == "PGD":
            run = 8
        else:
            exit()

        dataset_runs = [
            {"path": f"./project_data/experiment1/final_results/run{str(run)}/{dataset}/", "save_path": f"./project_data/experiment1/result_plots/{dataset}_{attack}/"}
        ]
        indexes = {"100": "(0, 100)", "200":"(0, 200)", "300":"(0, 300)", "500":"(0, 500)", "600":"(0, 600)", "700":"(0, 700)", "1000":"(0, 1000)", "1200":"(0, 1200)", "1500":"(0, 1500)", "2000":"(0, 2000)", "3000":"(0, 3000)", "5000":"(0, 5000)", "7000":"(0, 7000)", "10000": "(0, 10000)"}

        run = dataset_runs[0]

        path = run['path']
        save_plot_path = run['save_path']

        if not os.path.exists(save_plot_path):
            os.makedirs(save_plot_path)

        attack_list = {}

        att_names = {
            "LinfProjectedGradientDescentAttack": "PGDLinf",
            "L2ProjectedGradientDescentAttack": "PGDL2",
            "L2FastGradientAttack": "FGSML2",
            "LinfFastGradientAttack": "FGSMLinf"
        }
        named_labels = {}
        file_data = {}
        for f_name in natsorted(glob(f"{path}**/*.pickle")):
            if len(Path(f_name).stem.split("_")) > 1:
                continue

            version, key, norm = Path(f_name).stem.split(".")
            if int(key) > 2000:
                continue
            if norm == "inf":
                norm = str(np.inf)
            if version == "0":
                version = ""
            elif version == "1":
                version = "Straightened "

            net_name = version + Path(f_name).parent.name + norm
            named_labels[net_name] = version
            if net_name not in file_data:
                file_data[net_name] = {}
            
            with open(f_name, 'rb') as handle:
                file_data[net_name][key], experiment_settings = pickle.load(handle)

                attack, _ = str(experiment_settings['attack']).split('(')
                if attack in att_names:
                    attack = att_names[attack]

                attack_list[net_name] = attack

        if len(file_data) == 0:
            print("File data is empty")
            print(file_data)
            continue

        dataset_name = dataset.capitalize()
        with_clever = False

        comp = "completeness"
        all_file_data = {}
        if with_clever:
            clever_scores_means = {}

        for net_name, files in file_data.items():
            all_file_data[net_name] = {}
            if with_clever:
                clever_scores_means[net_name] = {}
            for k, data in files.items():
                if with_clever:
                    clever_scores_means[net_name][k] = {}
                for d, values in data.items():
                    if d not in all_file_data[net_name]:
                        all_file_data[net_name][d] = {}
                        
                    if comp not in all_file_data[net_name]:
                        all_file_data[net_name][comp] = {}

                    # if k in completeness_data:
                    #     all_file_data[net_name][comp][k] = completeness_data[k]

                    all_file_data[net_name][d][k] = values
                if with_clever:
                    if len(data['clever']) > 0 and type(data['clever']) == dict:
                        for key, val in data['clever'].items():
                            clever_scores_means[net_name][k][key] = np.mean(val)

                        clever_scores_means[net_name][k] = np.mean(val)

        key_emp = ['completeness', 'lsa_values', 'amise', 'amise_f1', 'emperical_robustness', 'emperical_robustness_logits']
        if with_clever:
            # keys = ['clever', 'equivalence_part', 'Centroid_part', 'bound_part', 'pairw_part', 'fol_L2', 'fol_Linf']
            keys = ['clever', 'equivalence_part', 'Centroid_part', 'fol_L2', 'fol_Linf']
        else:
            # keys = ['equivalence_part', 'Centroid_part', 'bound_part', 'pairw_part', 'fol_L2', 'fol_Linf']
            keys = ['equivalence_part', 'Centroid_part', 'fol_L2', 'fol_Linf']
        for net_name, files in all_file_data.items():
            for key in keys:
                files[f"{key}_"] = {}
                files[f"{key}_mean"] = {}
                for label, data in files[key].items():
                    if key in ['fol_L2', 'fol_Linf']:
                        # print(data)
                        files[f"{key}_"][label] = {}
                        for eps, d in data.items():
                            if len(d) == 0:
                                files[f"{key}_"][label][eps] = 0
                            else:
                                files[f"{key}_"][label][eps] = sum(d.values()) / len(d)
                        data = files[f"{key}_"][label]

                    if len(data) == 0:
                        files[f"{key}_mean"][label] = 0
                    else:
                        files[f"{key}_mean"][label] = sum(data.values()) / len(data)

        for net_name, files in all_file_data.items():
            for key in key_emp:
                _key = f"{key}_"
                __key = f"{key}__"
                files[_key] = {}
                files[__key] = {}
                files[f"{_key}mean"] = {}
                # label: data size, data: data
                for label, data in files[key].items():
                    # print(data.keys())
                    # k: Rob type,
                    for rob_type, data_per_eps in data.items():
                        if rob_type not in ['approx', 'diff', 'calc_diff', 'mean', 'all']:
                            continue

                        if rob_type not in files[_key]:
                            files[_key][rob_type] = {}
                            files[f"{_key}mean"][rob_type] = {}

                        if type(data_per_eps) == np.float64:
                            files[_key][rob_type][label] = data_per_eps
                            continue

                        if rob_type not in files[__key]:
                            files[__key][rob_type] = {}

                        if label not in files[_key][rob_type]:
                            files[_key][rob_type][label] = {}
                            files[f"{_key}mean"][rob_type][label] = {}

                        if type(data_per_eps) == list:
                            for eps, data_point in enumerate(data_per_eps):
                                if eps not in files[__key][rob_type]:
                                    files[__key][rob_type][eps] = {}

                                files[_key][rob_type][label][eps] = data_point
                                files[__key][rob_type][eps][label] = data_point
                        else:
                            for eps, data_point in data_per_eps.items():
                                if eps not in files[__key][rob_type]:
                                    files[__key][rob_type][eps] = {}

                                files[_key][rob_type][label][eps] = data_point
                                files[__key][rob_type][eps][label] = data_point

                        data = files[_key][rob_type][label]

                        if len(data) == 0:
                            files[f"{_key}mean"][rob_type][label] = 0
                        else:
                            files[f"{_key}mean"][rob_type][label] = sum(data.values()) / len(data)

        # keys = ['equivalence_part', 'Centroid_part', 'bound_part', 'pairw_part']
        keys = ['equivalence_part', 'Centroid_part']
        key_labels = {'equivalence_part': "Equivalence partitioning", "Centroid_part": 'Centroid partitioning'}

        for net_name, files in all_file_data.items():
            if "inf" in net_name:
                continue
            col = 0
            for key in keys:
                fig = plt.figure(figsize=(18, 5))
                gs = GridSpec(nrows=1, ncols=2)
                gs.update(wspace=0.1) # set the spacing between axes. 
                row = 0
                ax0 = fig.add_subplot(gs[row, col])
                lines = {0: []}
                for label, data in files[key].items():
                    x, y = zip(*sorted(data.items()))
                    
                    handles = ax0.plot(x, y, label=f"Input {indexes[label]}", marker='o')
                    lines[0].append(handles[0])
                    
                    ax0.set_title(f"{named_labels[net_name]}{key_labels[key]}")
                    ax0.set_xlabel('Classes')
                    ax0.set_ylabel('Data distribution')
                    # ax0.set_yscale('log')
                    ax0.autoscale(enable=True, axis='y')
                    if "Straightened" in net_name:
                        ax0.legend(handles=lines[0], loc=(0.0, 0.52), title=f"{dataset_name}")
                    elif col == 0:
                        ax0.legend(handles=lines[0], loc=(0.1, 0.51), title=f"{dataset_name}")
                    else:
                        ax0.legend(handles=lines[0], loc=(-0.01, 0.52), title=f"{dataset_name}")

                col += 1
            # row += 1

                fig.savefig(save_plot_path + f"data_coverage_line_indiv_{net_name}_{key_labels[key]}.jpg", bbox_inches='tight', transparent=False)

        plt.clf()
        
        # keys = ['equivalence_part', 'Centroid_part', 'bound_part', 'pairw_part']
        import seaborn as sns
        keys = ['equivalence_part', 'Centroid_part']
        key_labels = {'equivalence_part': "Equivalence partitioning", "Centroid_part": 'Centroid partitioning'}
        import matplotlib

        import pandas as pd


        for net_name, files in all_file_data.items():
            if "inf" in net_name:
                continue
            col = 0
            for key in keys:
                array = []
                xes = []
                yes = []
                for label, data in files[key].items():
                    arr = []
                    for x,y in data.items():
                        arr.append(y)
                    xes.append(np.array(arr))
                    yes.append(int(label))
                    
                grid_kws = {"height_ratios": (.88, .02), "hspace": .16}
                fig, (ax0, cbar_ax) = plt.subplots(2, figsize=(14, 9), gridspec_kw=grid_kws)
                row = 0
                lines = {0: []}
                x, y = zip(*sorted(files[key].items()))
                df = pd.DataFrame(data=xes, index=yes)
                sns.set_style("whitegrid")
                ax0 = sns.heatmap(df, 
                        ax=ax0,
                        cbar_ax=cbar_ax,
                        annot=False,
                        square=False,
                        cmap='inferno',
                        cbar_kws={ "orientation": "horizontal"})
                
                ax0.set_title(f"{named_labels[net_name]}{key_labels[key]}")
                ax0.set_xlabel('Classes')
                ax0.set_ylabel('Data inputs size')
                ax0.autoscale(enable=True, axis='y')

                fig.savefig(save_plot_path + f"data_coverage_heat_indiv_{net_name}_{key_labels[key]}.jpg", bbox_inches='tight', transparent=False)
        plt.clf()
        keys = ['lsa_values_']
        i = 0
        col = 0
        for net_name, files in all_file_data.items():
            if "inf" in net_name:
                continue
            lines = {0:[]}
            fig = plt.figure(figsize=(18, 5))
            gs = GridSpec(nrows=1, ncols=2)
            gs.update(wspace=0.1) # set the spacing between axes. 
            lines = {0: []}
            # fig = plt.figure(figsize=(9, 5))
            # gs = GridSpec(nrows=1, ncols=1)
            # gs.update(wspace=0.1) # set the spacing between axes. 
            # fig.suptitle(net_name)
            val = files["lsa_values_"]['mean']
            ax0 = fig.add_subplot(gs[0, col])
            x, y = zip(*natsorted(val.items()))
            # print(x)
            if type(y[0]) == dict:
                continue
            
            # ax0.plot(x, y, marker='o', c='tab:green')
            handles = ax0.plot(x, y, label=f"Input {indexes[label]}", marker='o')
            lines[0].append(handles[0])
            ax0.set_title(f"{named_labels[net_name]}{dataset_name} Distance-based surprise adequacy")
            
            ax0.set_xlabel('Data size')
            ax0.set_ylabel('Data surpisingness')

            # ax0.legend(handles=lines[0], loc=(0.785, 0.51), title=f"{dataset_name}")
            # ax0.grid(True)
            # ax0.set_xscale('log')
            # ax0.legend()
            col += 1
            fig.savefig(save_plot_path + f"DSA_plot_{net_name}.jpg", bbox_inches='tight', transparent=False)
            i += 1

        plt.clf()
        keys = ['emperical_robustness_']
        nrows = 1
        ncols = 2
        y_name = "F1"
        x_name = 'Epsilon'
        for net_name, files in all_file_data.items():
            # print(net_name)
            # if net_name != "No sampler basic_CNNinf":
            #     continue 
            fig = plt.figure(figsize=(18, 5))
            gs = GridSpec(nrows=1, ncols=2)
            gs.update(wspace=0.1) # set the spacing between axes. 
            for key in keys:
                col = 0
                row = 0
                lines = {0: [], 1: [], 2: []}
                ax0 = fig.add_subplot(gs[0, col])
                for type_, val in files[key].items():
                    # print(val)
                    if col % 3 == 0:
                        row += 1
                        col = 0
                    for label, data in val.items():
                        if type_ == 'approx':
                            handles1 = ax0.plot(*zip(*sorted(data.items())), label=f"AR {indexes[label]}", linestyle='dashed', marker='*')
                            lines[0].append(handles1[0])
                        else:
                            handles2 = ax0.plot(*zip(*sorted(data.items())), label=f"RR {indexes[label]}", marker='o')
                            lines[1].append(handles2[0])
                    col += 1
                ax0.set_title(f"Robustness " + attack_list[net_name])
                ax0.set_xlabel(x_name)
                ax0.set_ylabel(y_name)
                ax0.set_xscale('log')
                if dataset_name == "imagenet" or dataset_name == "Imagenet":
                    ax0.add_artist(ax0.legend(handles=lines[1], loc=(0.23, 0.), title=dataset_name))
                    ax0.legend(handles=lines[0], loc=(0.01,0.), title=dataset_name)
                else:
                    ax0.add_artist(ax0.legend(handles=lines[1], loc=(0.793, 0.35), title=dataset_name))
                    # first_legend = ax0.legend(handles=lines[0], loc=(1.04,0))
                    # ax0.add_artist(first_legend)
                    ax0.legend(handles=lines[0], loc=(0.575, 0.35), title=dataset_name)
                
                ax0.autoscale(enable=True, axis='y')

            fig.savefig(save_plot_path + f"approx_small_offset_calc_{net_name}.jpg", bbox_inches='tight', transparent=False)

        plt.clf()

        keys = ['amise_f1__']
        for net_name, files in all_file_data.items():
            # if net_name != "No sampler basic_CNNinf":
            #     continue 
            fig = plt.figure(figsize=(14, 9))
            gs = GridSpec(nrows=1, ncols=1)
            for key in keys:
                col = 0
                row = 0
                line1 = []
                line2 = []
                ax0 = fig.add_subplot(gs[0, 0])
                for type_ in reversed(files[key]):
                    val = files[key][type_]
                    # print(val)
                    if col % 3 == 0:
                        row += 1
                        col = 0
                    i_plots = 0
                    for label, data in val.items():
                        if type_ != 'approx':
                            if i_plots > 10:
                                handles2 = ax0.plot(*zip(*natsorted(data.items())), label=f"RR {round(label, 5)}", c=random.choice(all_colors), marker='o')
                            else:
                                handles2 = ax0.plot(*zip(*natsorted(data.items())), label=f"RR {round(label, 5)}", marker='o')

                            line1.append(handles2[0])
                        i_plots += 1
                        # else:    
                        #     handles1 = ax0.plot(*zip(*natsorted(data.items())), label=f" ", linestyle='dashed', marker='*')
                        #     line1.append(handles1[0])
                    col += 1

                ax0.set_title(f"{named_labels[net_name]} Completeness {attack}")
                ax0.set_xlabel('Size')
                ax0.set_ylabel('Completeness')
                ax0.set_yscale('log')
                
                ax0.legend(handles=line1, loc=(0.87, 0.4), title=dataset_name)#, ncol=2)
                
                ax0.autoscale(enable=True, axis='y')

            # fig.savefig(save_plot_path + f"amise_f1__{net_name}_OLD.jpg", bbox_inches='tight', transparent=False)
        plt.clf()
            
        keys = ['amise_f1_']
        nrows = 1
        ncols = 2

        x_name = 'Epsilon'
        y_name = 'AMISE'

        for net_name, files in all_file_data.items():
            fig = plt.figure(figsize=(28, 24))
            gs = GridSpec(nrows=3, ncols=2)
            for key in keys:
                col = 0
                row = 0
                ax0 = fig.add_subplot(gs[0, col])
                lines  = {0:[], 1:[]}
                for type_, val in files[key].items():
                    if col % 3 == 0:
                        row += 1
                        col = 0
                    for label, data in val.items():
                        for _k, val in data.items():
                            if val > 1000:
                                data[_k] = val / 1000000
                                
                        x, y = zip(*sorted(data.items()))
                            # print(val)
                        if type_ != 'approx':
                            handles2 = ax0.plot(x, y, label=f"RR {indexes[label]}", marker='o')
                            lines[0].append(handles2[0])
                        else:
                            handles1 = ax0.plot(*zip(*sorted(data.items())), label=f"AR {indexes[label]}", linestyle='dashed', marker='*')
                            lines[1].append(handles1[0])
                    col += 1
                # ax0.set_title(f"Robustness ({net_name})")
                ax0.set_title(f"{named_labels[net_name]} Completeness {attack}")
                # ax0.set_xlabel('Epsilon')
                # ax0.set_ylabel('AMISE')
                # ax0.set_xscale('log')
                # first_legend = ax0.legend(handles=line1, loc=(1.04,0))
                # ax0.add_artist(first_legend)
                # ax0.legend(handles=line2, loc=(1.05, 0.5))
                ax0.set_xlabel(x_name)
                ax0.set_ylabel(y_name)
                ax0.set_xscale('log')
                ax0.grid(False)
                ax0.add_artist(ax0.legend(handles=lines[0], loc=(0.01, 0.55), title=dataset))
                # first_legend = ax0.legend(handles=lines[0], loc=(1.04,0))
                # ax0.add_artist(first_legend)
                ax0.legend(handles=lines[1], loc=(0.155, 0.55), title=dataset)
                
                ax0.autoscale(enable=True, axis='y')
            fig.savefig(save_plot_path + f"amise_f1__{net_name}.jpg", bbox_inches='tight', transparent=False)

        plt.clf()