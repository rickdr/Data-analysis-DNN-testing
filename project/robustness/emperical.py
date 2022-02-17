import sys
import torch
import numpy as np
from scipy import stats
from sklearn import metrics

sys.path.insert(0, './')

from utils.map import _mAP

# def robustness(model, y, pred_y):
#     """
#     x: perturbed inputs, shape of x: [batch_size, width, height, channel] 
#     y: ground truth labels, shape of y: [batch_size] 
#     """
#     # res = torch.sum(torch.argmax(model(x), axis=1) == y) / y.shape[0]
#     res = torch.sum(y == pred_y) / y.shape[0]
#     return res.item()

# def emperical_robustness(data, targets, model):

#     robustness_list = []
#     for data, target in zip(data, targets):
#         robustness_list.append(robustness(model, data, target))

#     res = np.mean(torch.stack(robustness_list).cpu().numpy())
#     print("Overall robustness " + str(res))
#     return res.item()

def nantonum(values):
    return np.nan_to_num(values, posinf=1, neginf=0)

def _get_logits_diff(y_pred, y_true):
    """
    Returns the logits difference value for classification.
    :param x: adversarial inputs
    :param y: target outputs
    :return: the logits difference values as a torch tensor
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logit_correct = torch.gather(y_pred, index=y_true.unsqueeze(1), dim=1)

    logit_other = torch.Tensor()

    idx = torch.argsort(y_pred, dim=1)[:, -2].unsqueeze(1)
    inp = y_pred
    logit_diff = inp[~torch.zeros(inp.shape,dtype=torch.bool).to(device).scatter_(1, idx, torch.tensor(True).to(device).expand(idx.shape))].view(inp.size(0), inp.size(1) - idx.size(1))

    res = torch.mean(logit_diff, 1, True)[:, 0]
    if res.cuda:
        res = res.cpu()
    res = res.numpy()

    return res

# def _get_logits_diff(y_pred, y_true):
#     """
#     Returns the logits difference value for classification.
#     :param x: adversarial inputs
#     :param y: target outputs
#     :return: the logits difference values as a torch tensor
#     """
#     # for i in range(len(y_true)):
#     logit_correct = torch.gather(y_pred, index=y_true.unsqueeze(1), dim=1)
#     logit_diff = (y_pred - logit_correct)
#     logit_highest_incorrect = torch.gather(
#         y_pred, index=torch.argsort(y_pred, dim=1)[:, -2].unsqueeze(1), dim=1
#     )
#     res = (logit_correct - logit_highest_incorrect)[:, 0]
#     # logit_diff_mean = torch.mean(logit_diff, 1, True)[:, 0]
#     return res


# def _get_logits_diff(y_pred, y_true):
#     """
#     Returns the logits difference value for classification.
#     :param x: adversarial inputs
#     :param y: target outputs
#     :return: the logits difference values as a torch tensor
#     """
#     logit_correct = torch.gather(y_pred, index=y_true.unsqueeze(1), dim=1)
#     res = (1 - logit_correct)
#     return res


def calc_correlations(emp_values):
    avg_accuracy = "micro"
    emp_robs = {"approx": {}, "diff": {},"calc": {},\
        "approx_corr_rob": {}, "tau_approx_corr_rob": {},\
        "spear_approx_corr_rob": {}, "approx_corr_diff": {},\
        "tau_approx_corr_diff": {}, "spear_approx_corr_diff": {},\
        "approx_pvalue": {}, "tau_approx_pvalue": {}, "spear_approx_pvalue": {},\
        "approx_diff_pvalue": {}, "tau_approx_diff_pvalue": {}, "spear_approx_diff_pvalue": {}}

    (_, _approx), (_, _calc), (_, _calc_diff) = emp_values

    for eps in _approx.keys():
        approx_ = _approx[eps]
        calc_ = _calc[eps]
        calc_diff_ = _calc_diff[eps]
        # emp_values[0]
        if len(approx_) > 1:
            approx_ = nantonum(approx_)
            calc_ = nantonum(calc_)
            calc_diff_ = nantonum(calc_diff_)

            approx_corr_rob, approx_pvalue = stats.pearsonr(approx_, calc_)
            tau_approx_corr_rob, tau_approx_pvalue = stats.kendalltau(approx_, calc_)
            spear_approx_corr_rob, spear_approx_pvalue = stats.spearmanr(approx_, calc_)

            approx_corr_diff, approx_diff_pvalue = stats.pearsonr(approx_, calc_diff_)
            tau_approx_corr_diff, tau_approx_diff_pvalue = stats.kendalltau(approx_, calc_diff_)
            spear_approx_corr_diff, spear_approx_diff_pvalue = stats.spearmanr(approx_, calc_diff_)
        else:
            approx_corr_rob = np.nan
            approx_corr_diff = np.nan

            approx_pvalue = np.nan
            approx_diff_pvalue = np.nan

        emp_robs["approx"][eps] = np.mean(approx_)
        emp_robs["diff"][eps] = np.mean(calc_diff_)
        emp_robs["calc"][eps] = np.mean(calc_)

        if np.isnan(approx_corr_rob):
            approx_corr_rob = 0
            tau_approx_corr_rob = 0
            spear_approx_corr_rob = 0

        emp_robs["approx_corr_rob"][eps] = approx_corr_rob
        emp_robs["tau_approx_corr_rob"][eps] = tau_approx_corr_rob
        emp_robs["spear_approx_corr_rob"][eps] = spear_approx_corr_rob
        
        if np.isnan(approx_corr_diff):
            approx_corr_diff = 0
            tau_approx_corr_diff = 0
            spear_approx_corr_diff = 0

        emp_robs["approx_corr_diff"][eps] = approx_corr_diff
        emp_robs["tau_approx_corr_diff"][eps] = tau_approx_corr_diff
        emp_robs["spear_approx_corr_diff"][eps] = spear_approx_corr_diff
        
        if np.isnan(approx_pvalue):
            approx_pvalue = 0
            tau_approx_pvalue = 0
            spear_approx_pvalue = 0
        
        emp_robs["approx_pvalue"][eps] = approx_pvalue
        emp_robs["tau_approx_pvalue"][eps] = tau_approx_pvalue
        emp_robs["spear_approx_pvalue"][eps] = spear_approx_pvalue
        
        if np.isnan(approx_diff_pvalue):
            approx_diff_pvalue = 0
            tau_approx_diff_pvalue = 0
            spear_approx_diff_pvalue = 0

        emp_robs["approx_diff_pvalue"][eps] = approx_diff_pvalue
        emp_robs["tau_approx_diff_pvalue"][eps] = tau_approx_diff_pvalue
        emp_robs["spear_approx_diff_pvalue"][eps] = spear_approx_diff_pvalue

    return emp_robs



def calculate_logit(fmodel, dataset, clipped, targets, epsilons):
    z_results = {}
    approx = {}
    calc_diff = {}
    calc = {}
    for key, eps in enumerate(epsilons):
        approx_ = []
        calc_diff_ = []
        calc_ = []
        for i in range(len(clipped)):
            y = targets[i]
            y_pred = fmodel(dataset[i])
            y_pred_adv = fmodel(clipped[i][key])

            logit_benign = _get_logits_diff(y_true=y, y_pred=y_pred) # Benign
            logit_attacked = _get_logits_diff(y_true=y, y_pred=y_pred_adv) # Attacked
            logit_approx = _get_logits_diff(y_true=torch.argmax(y_pred, axis=1), y_pred=y_pred_adv) # Benign / attacked

            logit_benign = nantonum(logit_benign)
            logit_attacked = nantonum(logit_attacked)
            logit_approx = nantonum(logit_approx)

            if all(v == 0. for v in logit_benign):
                logit_benign = [v + np.random.uniform(1e-10, 1e-7) for v in logit_benign]

            if all(v == 0. for v in logit_attacked):
                logit_attacked = [v + np.random.uniform(1e-10, 1e-7) for v in logit_attacked]

            if all(v == 0. for v in logit_approx):
                logit_approx = [v + np.random.uniform(1e-10, 1e-7) for v in logit_approx]

            if all(v > 1. for v in logit_benign):
                logit_benign = [1 - np.random.uniform(1e-10, 1e-9) for v in logit_benign]

            if all(v > 1. for v in logit_attacked):
                logit_attacked = [1 - np.random.uniform(1e-10, 1e-9) for v in logit_attacked]

            if all(v > 1. for v in logit_approx):
                logit_approx = [1 - np.random.uniform(1e-10, 1e-9) for v in logit_approx]

            if all(v == 1. for v in logit_benign):
                logit_benign = [v - np.random.uniform(1e-10, 1e-8) for v in logit_benign]

            if all(v == 1. for v in logit_attacked):
                logit_attacked = [v - np.random.uniform(1e-10, 1e-8) for v in logit_attacked]

            if all(v == 1. for v in logit_approx):
                logit_approx = [v - np.random.uniform(1e-10, 1e-8) for v in logit_approx]

            logit_diff = logit_attacked / logit_benign
            logit_diff = nantonum(logit_diff)

            if all(v == 0. for v in logit_diff):
                logit_diff = [v + np.random.uniform(1e-10, 1e-7) for v in logit_diff]

            if all(v > 1. for v in logit_diff):
                logit_diff = [1 - np.random.uniform(1e-10, 1e-9) for v in logit_diff]

            if all(v == 1. for v in logit_diff):
                logit_diff = [v - np.random.uniform(1e-10, 1e-8) for v in logit_diff]

            approx_.append(logit_approx)
            calc_diff_.append(logit_diff)
            calc_.append(logit_attacked)

        approx_ = np.array([item for sublist in approx_ for item in sublist])
        calc_diff_ = np.array([item for sublist in calc_diff_ for item in sublist])
        calc_ = np.array([item for sublist in calc_ for item in sublist])

        approx[eps] = approx_
        calc_diff[eps] = calc_diff_
        calc[eps] = calc_

    emp_results = [("approx", approx), ("calc_diff", calc_diff), ("calc", calc)]
    z_results = calc_correlations(emp_results)

    return emp_results, z_results


def calculate_f1(fmodel, dataset, clipped, targets, epsilons):
    avg_accuracy = "micro"
    z_results = {}
    approx = {}
    calc_diff = {}
    calc = {}
    for key, eps in enumerate(epsilons):
        approx_ = []
        calc_diff_ = []
        calc_ = []
        for i in range(len(clipped)):
            y = targets[i]
            y_pred = torch.argmax(fmodel(dataset[i]), axis=1)
            y_pred_adv = torch.argmax(fmodel(clipped[i][key]), axis=1)
            if y.cuda:
                y = y.cpu()
            if y.cuda:
                y_pred = y_pred.cpu()
            if y.cuda:
                y_pred_adv = y_pred_adv.cpu()

            y = y.numpy()
            y_pred = y_pred.numpy()
            y_pred_adv = y_pred_adv.numpy()

            f1_benign = metrics.f1_score(y_true=y, y_pred=y_pred, average=avg_accuracy)
            f1_attacked = metrics.f1_score(y_true=y, y_pred=y_pred_adv, average=avg_accuracy)
            f1_approx = metrics.f1_score(y_true=y_pred, y_pred=y_pred_adv, average=avg_accuracy)

            if np.isnan(f1_attacked) or np.isnan(f1_benign) or np.isnan(f1_approx):
                print("Possible nans", f1_attacked, f1_benign, f1_approx)

            f1_benign = nantonum(f1_benign)
            f1_attacked = nantonum(f1_attacked)
            f1_approx = nantonum(f1_approx)
            
            f1_diff = f1_attacked / f1_benign
            f1_diff = nantonum(f1_diff)

            approx_.append(f1_approx)
            calc_diff_.append(f1_diff)
            calc_.append(f1_attacked)
        
        if all(v == 0. for v in approx_):
            approx_ = [v + np.random.uniform(1e-10, 1e-7) for v in approx_]

        if all(v == 0. for v in calc_):
            calc_ = [v + np.random.uniform(1e-10, 1e-7) for v in calc_]

        if all(v == 0. for v in calc_diff_):
            calc_diff_ = [v + np.random.uniform(1e-10, 1e-7) for v in calc_diff_]

        if all(v == 1.0 for v in approx_):
            approx_ = [v - np.random.uniform(1e-10, 1e-8) for v in approx_]

        if all(v == 1.0 for v in calc_):
            calc_ = [v - np.random.uniform(1e-10, 1e-8) for v in calc_]

        if all(v == 1.0 for v in calc_diff_):
            calc_diff_ = [v - np.random.uniform(1e-10, 1e-8) for v in calc_diff_]

        if all(v > 1. for v in approx_):
            approx_ = [1 - np.random.uniform(1e-10, 1e-9) for v in approx_]

        if all(v > 1. for v in calc_):
            calc_ = [1 - np.random.uniform(1e-10, 1e-9) for v in calc_]

        if all(v > 1. for v in calc_diff_):
            calc_diff_ = [1 - np.random.uniform(1e-10, 1e-9) for v in calc_diff_]

        approx[eps] = approx_
        calc_diff[eps] = calc_diff_
        calc[eps] = calc_

    emp_results = [("approx", approx), ("calc_diff", calc_diff), ("calc", calc)]
    z_results = calc_correlations(emp_results)

    return emp_results, z_results


# def calculate(fmodel, dataset, clipped, targets, epsilons):

#     for key, eps in enumerate(epsilons):
#         emp_robs["approx"][eps] = []
#         emp_robs["diff"][eps] = []
#         emp_robs["calc"][eps] = []

#         for i in range(len(clipped)):
#             y = targets[i]
#             y_pred = torch.argmax(fmodel(dataset[i]), axis=1)
#             y_pred_adv = torch.argmax(fmodel(clipped[i][key]), axis=1)

#             y = y.cpu().numpy()
#             y_pred = y_pred.cpu().numpy()
#             y_pred_adv = y_pred_adv.cpu().numpy()

#             f1_benign = metrics.f1_score(y_true=y, y_pred=y_pred, average=avg_accuracy)
#             f1_attacked = metrics.f1_score(y_true=y, y_pred=y_pred_adv, average=avg_accuracy)
#             f1_approx = metrics.f1_score(y_true=y_pred, y_pred=y_pred_adv, average=avg_accuracy)

#             f1_diff = f1_attacked / f1_benign

#             emp_robs["approx"][eps].append(f1_approx)
#             emp_robs["diff"][eps].append(f1_diff)
#             emp_robs["calc"][eps].append(f1_attacked)
        
#         approx_ = emp_robs["approx"][eps]
#         calc_diff_ = emp_robs["diff"][eps]
#         calc_ = emp_robs["calc"][eps]

#         emp_all = dict(approx= approx_, diff= calc_diff_, calc= calc_)

#         if len(approx_) > 1:
#             if all(v == 0 for v in approx_):
#                 approx_ = [v + np.random.uniform(1e-10, 1e-7) for v in approx_]

#             if all(v == 0 for v in calc_):
#                 calc_ = [v + np.random.uniform(1e-10, 1e-7) for v in calc_]

#             if all(v == 0 for v in calc_diff_):
#                 calc_diff_ = [v + np.random.uniform(1e-10, 1e-7) for v in calc_diff_]            

#             approx_corr_rob, approx_pvalue = stats.pearsonr(approx_, calc_)
#             tau_approx_corr_rob, tau_approx_pvalue = stats.kendalltau(approx_, calc_)
#             spear_approx_corr_rob, spear_approx_pvalue = stats.spearmanr(approx_, calc_)

#             approx_corr_diff, approx_diff_pvalue = stats.pearsonr(approx_, calc_diff_)
#             tau_approx_corr_diff, tau_approx_diff_pvalue = stats.kendalltau(approx_, calc_diff_)
#             spear_approx_corr_diff, spear_approx_diff_pvalue = stats.spearmanr(approx_, calc_diff_)
#         else:
#             approx_corr_rob = np.nan
#             approx_corr_diff = np.nan

#             approx_pvalue = np.nan
#             approx_diff_pvalue = np.nan

#         emp_robs["approx"][eps] = np.mean(approx_)
#         emp_robs["diff"][eps] = np.mean(calc_diff_)
#         emp_robs["calc"][eps] = np.mean(calc_)

#         if np.isnan(approx_corr_rob):
#             approx_corr_rob = 0
#             tau_approx_corr_rob = 0
#             spear_approx_corr_rob = 0

#         emp_robs["approx_corr_rob"][eps] = approx_corr_rob
#         emp_robs["tau_approx_corr_rob"][eps] = tau_approx_corr_rob
#         emp_robs["spear_approx_corr_rob"][eps] = spear_approx_corr_rob
        
#         if np.isnan(approx_corr_diff):
#             approx_corr_diff = 0
#             tau_approx_corr_diff = 0
#             spear_approx_corr_diff = 0

#         emp_robs["approx_corr_diff"][eps] = approx_corr_diff
#         emp_robs["tau_approx_corr_diff"][eps] = tau_approx_corr_diff
#         emp_robs["spear_approx_corr_diff"][eps] = spear_approx_corr_diff
        
#         if np.isnan(approx_pvalue):
#             approx_pvalue = 0
#             tau_approx_pvalue = 0
#             spear_approx_pvalue = 0
        
#         emp_robs["approx_pvalue"][eps] = approx_pvalue
#         emp_robs["tau_approx_pvalue"][eps] = tau_approx_pvalue
#         emp_robs["spear_approx_pvalue"][eps] = spear_approx_pvalue
        
#         if np.isnan(approx_diff_pvalue):
#             approx_diff_pvalue = 0
#             tau_approx_diff_pvalue = 0
#             spear_approx_diff_pvalue = 0

#         emp_robs["approx_diff_pvalue"][eps] = approx_diff_pvalue
#         emp_robs["tau_approx_diff_pvalue"][eps] = tau_approx_diff_pvalue
#         emp_robs["spear_approx_diff_pvalue"][eps] = spear_approx_diff_pvalue

#     return emp_all, emp_robs


# def calculate_logit(fmodel, dataset, clipped, targets, epsilons):
#     avg_accuracy = "micro"
#     emp_robs = {"approx": {}, "diff": {},"calc": {},\
#         "approx_corr_rob": {}, "tau_approx_corr_rob": {},\
#         "spear_approx_corr_rob": {}, "approx_corr_diff": {},\
#         "tau_approx_corr_diff": {}, "spear_approx_corr_diff": {},\
#         "approx_pvalue": {}, "tau_approx_pvalue": {}, "spear_approx_pvalue": {},\
#         "approx_diff_pvalue": {}, "tau_approx_diff_pvalue": {}, "spear_approx_diff_pvalue": {}}

#     for key, eps in enumerate(epsilons):
#         approx_ = []
#         calc_diff_ = []
#         calc_ = []
#         for i in range(len(clipped)):
#             y = targets[i]
#             y_pred = fmodel(dataset[i])
#             y_pred_adv = fmodel(clipped[i][key])

#             f1_benign = _get_logits_diff(y_true=y, y_pred=y_pred) # Benign
#             f1_attacked = _get_logits_diff(y_true=y, y_pred=y_pred_adv) # Attacked
#             f1_approx = _get_logits_diff(y_true=torch.argmax(y_pred, axis=1), y_pred=y_pred_adv) # Benign / attacked

#             f1_diff = f1_attacked / f1_benign
#             f1_benign = f1_benign.cpu().numpy()
#             f1_attacked = f1_attacked.cpu().numpy()
#             f1_approx = f1_approx.cpu().numpy()

#             f1_diff = f1_diff.cpu().numpy()
#             f1_diff[np.isnan(f1_diff)] = 0.
#             f1_diff[f1_diff == -np.inf] = 0.
#             f1_diff[f1_diff == np.inf] = 0.

#             approx_.append(f1_approx)
#             calc_diff_.append(f1_diff)
#             calc_.append(f1_attacked)

#         approx_ = np.array([item for sublist in approx_ for item in sublist])
#         calc_diff_ = np.array([item for sublist in calc_diff_ for item in sublist])
#         calc_ = np.array([item for sublist in calc_ for item in sublist])

#         emp_all = dict(approx=approx_, diff=calc_diff_, calc=calc_)

#         if len(approx_) > 1:
#             if all(v == 0 for v in approx_):
#                 approx_ = [v + np.random.uniform(1e-10, 1e-7) for v in approx_]

#             if all(v == 0 for v in calc_):
#                 calc_ = [v + np.random.uniform(1e-10, 1e-7) for v in calc_]

#             if all(v == 0 for v in calc_diff_):
#                 calc_diff_ = [v + np.random.uniform(1e-10, 1e-7) for v in calc_diff_]
                
#             approx_corr_rob, approx_pvalue = stats.pearsonr(approx_, calc_)
#             tau_approx_corr_rob, tau_approx_pvalue = stats.kendalltau(approx_, calc_)
#             spear_approx_corr_rob, spear_approx_pvalue = stats.spearmanr(approx_, calc_)

#             approx_corr_diff, approx_diff_pvalue = stats.pearsonr(approx_, calc_diff_)
#             tau_approx_corr_diff, tau_approx_diff_pvalue = stats.kendalltau(approx_, calc_diff_)
#             spear_approx_corr_diff, spear_approx_diff_pvalue = stats.spearmanr(approx_, calc_diff_)
#         else:
#             approx_corr_rob = np.nan
#             approx_corr_diff = np.nan

#             approx_pvalue = np.nan
#             approx_diff_pvalue = np.nan

#         emp_robs["approx"][eps] = np.mean(approx_)
#         emp_robs["diff"][eps] = np.mean(calc_diff_)
#         emp_robs["calc"][eps] = np.mean(calc_)

#         if np.isnan(approx_corr_rob):
#             approx_corr_rob = 0
#             tau_approx_corr_rob = 0
#             spear_approx_corr_rob = 0

#         emp_robs["approx_corr_rob"][eps] = approx_corr_rob
#         emp_robs["tau_approx_corr_rob"][eps] = tau_approx_corr_rob
#         emp_robs["spear_approx_corr_rob"][eps] = spear_approx_corr_rob
        
#         if np.isnan(approx_corr_diff):
#             approx_corr_diff = 0
#             tau_approx_corr_diff = 0
#             spear_approx_corr_diff = 0

#         emp_robs["approx_corr_diff"][eps] = approx_corr_diff
#         emp_robs["tau_approx_corr_diff"][eps] = tau_approx_corr_diff
#         emp_robs["spear_approx_corr_diff"][eps] = spear_approx_corr_diff
        
#         if np.isnan(approx_pvalue):
#             approx_pvalue = 0
#             tau_approx_pvalue = 0
#             spear_approx_pvalue = 0
        
#         emp_robs["approx_pvalue"][eps] = approx_pvalue
#         emp_robs["tau_approx_pvalue"][eps] = tau_approx_pvalue
#         emp_robs["spear_approx_pvalue"][eps] = spear_approx_pvalue
        
#         if np.isnan(approx_diff_pvalue):
#             approx_diff_pvalue = 0
#             tau_approx_diff_pvalue = 0
#             spear_approx_diff_pvalue = 0

#         emp_robs["approx_diff_pvalue"][eps] = approx_diff_pvalue
#         emp_robs["tau_approx_diff_pvalue"][eps] = tau_approx_diff_pvalue
#         emp_robs["spear_approx_diff_pvalue"][eps] = spear_approx_diff_pvalue

#     return emp_robs


# def calculate_detection(model, dataset, clipped, targets, epsilons):
#     iou_threshold = 0.5
#     box_format = "corners"
#     num_classes = 91
#     method = "interp"

#     emp_robs = {"approx": {}, "diff": {},"calc": {},\
#         "approx_corr_rob": {}, "tau_approx_corr_rob": {},\
#         "spear_approx_corr_rob": {}, "approx_corr_diff": {},\
#         "tau_approx_corr_diff": {}, "spear_approx_corr_diff": {},\
#         "approx_pvalue": {}, "tau_approx_pvalue": {}, "spear_approx_pvalue": {},\
#         "approx_diff_pvalue": {}, "tau_approx_diff_pvalue": {}, "spear_approx_diff_pvalue": {}}

#     for key, eps in enumerate(epsilons):
#         emp_robs["approx"][eps] = []
#         emp_robs["diff"][eps] = []
#         emp_robs["calc"][eps] = []
        
#         emp_robs["approx_corr_rob"][eps] = []
#         emp_robs["tau_approx_corr_rob"][eps] = []
#         emp_robs["spear_approx_corr_rob"][eps] = []

#         emp_robs["approx_corr_diff"][eps] = []
#         emp_robs["tau_approx_corr_diff"][eps] = []
#         emp_robs["spear_approx_corr_diff"][eps] = []
        
#         emp_robs["approx_pvalue"][eps] = []
#         emp_robs["tau_approx_pvalue"][eps] = []
#         emp_robs["spear_approx_pvalue"][eps] = []

#         emp_robs["approx_diff_pvalue"][eps] = []
#         emp_robs["tau_approx_diff_pvalue"][eps] = []
#         emp_robs["spear_approx_diff_pvalue"][eps] = []

#         for i in range(len(dataset)):
#             map_benign_ = []
#             map_attacked_ = []
#             map_approx_ = []
#             y = targets[i]
#             y_pred = model(dataset[i])
#             y_pred_adv = model(clipped[key][i])

#             # for j in range(len(dataset[i])):
#             #     y_ = y[j]
#             #     y_pred_ = y_pred[j]['boxes'].detach().cpu().numpy()
#             #     y_pred_adv_ = y_pred_adv[j]['boxes'].detach().cpu().numpy()

#             map_benign_.append(_mAP(y_pred, y, iou_threshold=iou_threshold, box_format=box_format, num_classes=num_classes, method=method))
#             map_benign_.append(_mAP(y, y, iou_threshold=iou_threshold, box_format=box_format, num_classes=num_classes, method=method))
#             map_attacked_.append(_mAP(y_pred_adv, y, iou_threshold=iou_threshold, box_format=box_format, num_classes=num_classes, method=method))
#             map_approx_.append(_mAP(y_pred_adv, y_pred, iou_threshold=iou_threshold, box_format=box_format, num_classes=num_classes, method=method))

#             emp_robs["approx"][eps].append(np.mean(map_approx_))
#             emp_robs["diff"][eps].append(np.mean(map_attacked_) / np.mean(map_benign_))
#             emp_robs["calc"][eps].append(np.mean(map_attacked_))
        
#         approx_ = emp_robs["approx"][eps]
#         calc_diff_ = emp_robs["diff"][eps]
#         calc_ = emp_robs["calc"][eps]

#         if len(approx_) > 1:
#             if all(v == 0 for v in approx_):
#                 approx_ = [v + np.random.uniform(1e-10, 1e-7) for v in approx_]

#             if all(v == 0 for v in calc_):
#                 calc_ = [v + np.random.uniform(1e-10, 1e-7) for v in calc_]

#             if all(v == 0 for v in calc_diff_):
#                 calc_diff_ = [v + np.random.uniform(1e-10, 1e-7) for v in calc_diff_]            

#             approx_corr_rob, approx_pvalue = stats.pearsonr(approx_, calc_)
#             tau_approx_corr_rob, tau_approx_pvalue = stats.kendalltau(approx_, calc_)
#             spear_approx_corr_rob, spear_approx_pvalue = stats.spearmanr(approx_, calc_)

#             approx_corr_diff, approx_diff_pvalue = stats.pearsonr(approx_, calc_diff_)
#             tau_approx_corr_diff, tau_approx_diff_pvalue = stats.kendalltau(approx_, calc_diff_)
#             spear_approx_corr_diff, spear_approx_diff_pvalue = stats.spearmanr(approx_, calc_diff_)
#         else:
#             approx_corr_rob = np.nan
#             approx_corr_diff = np.nan

#             approx_pvalue = np.nan
#             approx_diff_pvalue = np.nan

#         emp_robs["approx"][eps] = np.mean(approx_)
#         emp_robs["diff"][eps] = np.mean(calc_diff_)
#         emp_robs["calc"][eps] = np.mean(calc_)

#         if np.isnan(approx_corr_rob):
#             approx_corr_rob = 0
#             tau_approx_corr_rob = 0
#             spear_approx_corr_rob = 0

#         emp_robs["approx_corr_rob"][eps] = approx_corr_rob
#         emp_robs["tau_approx_corr_rob"][eps] = tau_approx_corr_rob
#         emp_robs["spear_approx_corr_rob"][eps] = spear_approx_corr_rob
        
#         if np.isnan(approx_corr_diff):
#             approx_corr_diff = 0
#             tau_approx_corr_diff = 0
#             spear_approx_corr_diff = 0

#         emp_robs["approx_corr_diff"][eps] = approx_corr_diff
#         emp_robs["tau_approx_corr_diff"][eps] = tau_approx_corr_diff
#         emp_robs["spear_approx_corr_diff"][eps] = spear_approx_corr_diff
        
#         if np.isnan(approx_pvalue):
#             approx_pvalue = 0
#             tau_approx_pvalue = 0
#             spear_approx_pvalue = 0
        
#         emp_robs["approx_pvalue"][eps] = approx_pvalue
#         emp_robs["tau_approx_pvalue"][eps] = tau_approx_pvalue
#         emp_robs["spear_approx_pvalue"][eps] = spear_approx_pvalue
        
#         if np.isnan(approx_diff_pvalue):
#             approx_diff_pvalue = 0
#             tau_approx_diff_pvalue = 0
#             spear_approx_diff_pvalue = 0

#         emp_robs["approx_diff_pvalue"][eps] = approx_diff_pvalue
#         emp_robs["tau_approx_diff_pvalue"][eps] = tau_approx_diff_pvalue
#         emp_robs["spear_approx_diff_pvalue"][eps] = spear_approx_diff_pvalue

#     return emp_robs