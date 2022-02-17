import torch
from matplotlib import pyplot as plt
import numpy as np

from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# def eq_part(dataset) -> float:
#     """number of testcases per class * total number of classes in dataset
#     --------------------------------
#     total number of testcases"""

#     all_classes = dataset.classes
#     num_classes = len(all_classes)

#     examples = {i: 0 for i in range(num_classes)}
#     amounts = {i: 0 for i in range(num_classes)}

#     for x, i in dataset:
#         examples[i] += 1

#     total = sum(examples.values())
#     for x, i in enumerate(examples):
#         amounts[i] = examples[i] * num_classes / total

#     # calulate mean
#     # mean_dist = float(sum(amounts.values())) / num_classes
#     minimal_dist = float(min(amounts.values())) / num_classes

#     return minimal_dist

def num_per_target(targets):
    target_count = {}
    for target in targets:
        for key in target:
            if hasattr(key, 'cpu'):
                key = key.cpu()
            key = key.item()

            if key not in target_count:
                target_count[key] = 0
            target_count[key] += 1
    return target_count

def eq_part(targets):
    target_count = num_per_target(targets)
    num_classes = len(target_count)
    means = {}
    total = sum(target_count.values())
    for x, i in enumerate(target_count):
        means[i] = target_count[i] * num_classes / total
    # mean_dist = float(sum(means.values())) / num_classes
    # minimal_dist = float(min(means.values())) / num_classes
    return means    

def cent_pos(dataset, targets, bound_threshold=10):
    """
    for case in range(classes)
        Average euclidean distance centroid to test points per class
        --------------------------------
        number of testcases per class
    """
    class_sorted_data = {}
    centroids = {}
    target_count = num_per_target(targets)
    
    # sort points per class
    for data, target in zip(dataset, targets):
        for data_point, label in zip(data, target):
            if label.item() not in class_sorted_data:
                class_sorted_data[label.item()] = []

            point = data_point.reshape(-1)
            class_sorted_data[label.item()].append(point)

    for target, data in class_sorted_data.items():
        # Calculate centroid(mean of points per target)
        mean_example = torch.mean(torch.stack(data), dim=0)
        centroids[target] = mean_example

        dists = []
        for i, data_point in enumerate(data):
            dist = torch.linalg.norm(centroids[target] - data_point)
            dists.append(1 if dist.item() <= bound_threshold else 0)

        class_sorted_data[target] = sum(dists) / target_count[target]

    return class_sorted_data


def bound_cond(fmodel, dataset, targets, bound_threshold=(-30, 30)):
    target_distance = {}
    class_sorted_data = {}
    centroids = {}
    target_count = num_per_target(targets)
    
    # sort points per class
    for data, target in zip(dataset, targets):
        batch_data = []
        for data_point, label in zip(data, target):
            if label.item() not in class_sorted_data:
                class_sorted_data[label.item()] = []

            batch_data.append(data_point)

        class_sorted_data[label.item()].append(torch.stack(batch_data))

    for target, data in class_sorted_data.items():
        decisions = []
        for i, data_points in enumerate(data):
            for point in fmodel(data_points).cpu().numpy():
                decision_val = point[target].item()
                decisions.append(0 if bound_threshold[0] < decision_val \
                    and decision_val > bound_threshold[1] else 1)

        class_sorted_data[target] = sum(decisions) / target_count[target]

    # res = float(sum(target_distance.values())) / len(target_distance)
    return class_sorted_data


def pairw_cond(fmodel, dataset, targets, bound_threshold=(-20, 20)):
    class_sorted_data = {}
    class_dists = {}
    target_count = num_per_target(targets)

    # sort points per class
    for data, target in zip(dataset, targets):
        for data_point, label in zip(data, target):
            if label.item() not in class_sorted_data:
                class_sorted_data[label.item()] = []
                class_dists[label.item()] = []

            class_sorted_data[label.item()].append(data)

    for target, data in class_sorted_data.items():
        decisions = []
        for datapoints in data:
            pred = fmodel(datapoints)
            for i, point in enumerate(pred.cpu().numpy()):
                # for labels in point:
                if target == i:
                    continue

                decision_val = point[target].item()
                class_dists[target].append(0 if bound_threshold[0] < decision_val \
                    and decision_val > bound_threshold[1] else 1)

    for target, dist in class_dists.items():
        class_sorted_data[target] = sum(dist) / len(class_dists[target])

    # res = float(sum(target_distance.values())) / len(target_distance)
    return class_sorted_data


# def Pairw_bound_cond(svm, dataset, targets):
#     target_distance = {}
#     for target, data in zip(targets, dataset):
#         datapoints = data.cpu().numpy().reshape(len(target), -1)
#         decision = svm.predict(datapoints)

#         for index, decision in zip(target, decision):
#             key = index.item()
#             if key not in target_distance:
#                 target_distance[key] = []

#             target_distance[key].append(1 if key == decision else 0)

#     for target, data in target_distance.items():
#         target_distance[target] = sum(data) / len(data)

#     res = float(sum(target_distance.values())) / len(target_distance)
#     return res

# def Pairw_bound_cond(dataset, targets):
#     if svm is None:
#         return "run Bounded conditioning first to obtain a SVM"

#     support_indices = np.cumsum(svm.n_support_)

#     unique_targets = []
#     for target in targets:
#         for j in target:
#             key = j.cpu().item()
#             if key not in unique_targets:
#                 unique_targets.append(key)

#     result = {svm.dual_coef_[i:support_indices[0]] for i in unique_targets}

#     return svm.dual_coef_

# def bound_cond(dataset) -> float:
#     """    
#     for case in range(classes)
#         Average distance to classification bound per class
#         --------------------------------
#         number of testcases per class
#     """

#     class_sorted_data = {}
#     for data, target in data_loader:
#         for data_point, label in zip(data, target):
#             if label.item() not in class_sorted_data:
#                 class_sorted_data[label.item()] = []

#             point = data_point.reshape(-1)
#             class_sorted_data[label.item()].append(point)

#     # Init centroids
#     centroids = {}
#     for label, data in class_sorted_data.items():
#         mean_example = torch.mean(torch.stack(data), dim=0)
#         centroids[label] = mean_example / len(data)


# def Pairw_bound_cond(dataset) -> float:
#     for case in range(classes)
#         Average distance to classification bound per class
#         --------------------------------
#         number of testcases per class

#     Test each class against each other
