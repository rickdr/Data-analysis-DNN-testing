import torch
from matplotlib import pyplot as plt
import numpy as np

from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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