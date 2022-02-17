import sys
import torch
import numpy as np
# import cupy as cp
from tqdm import tqdm
from scipy import stats
from sklearn import metrics
import matplotlib.pyplot as plt
import concurrent
from concurrent.futures.thread import ThreadPoolExecutor

sys.path.insert(0, '../')

from project.completeness import KDE
from project.robustness import emperical

        # with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        #     while start < num_targets:

        #         # Select batch
        #         diff = num_targets - start
        #         if diff < self.dsa_batch_size:
        #             batch = target_pred[start:start + diff]
        #         else:
        #             batch = target_pred[start: start + self.dsa_batch_size]

        #         # Calculate DSA per label
        #         for label in range(self.config.num_classes):

        #             def task(t_batch, t_label, t_start):
        #                 matches = np.where(t_batch == t_label)
        #                 if len(matches[0]) > 0:
        #                     a_min_dist, b_min_dist = self._dsa_distances(t_label, matches, t_start, target_ats)
        #                     t_task_dsa = a_min_dist / b_min_dist
        #                     return matches[0], t_start, t_task_dsa
        #                 else:
        #                     return None, None, None

        #             futures.append(executor.submit(task, np.copy(batch), label, start))

        #         start += self.dsa_batch_size

        # for future in futures:
        #     f_idxs, f_start, f_task_dsa = future.result()
        #     if f_idxs is not None:
        #         dsa[f_idxs + f_start] = f_task_dsa


# def __calc_kde(emp_values):
#     z_results = {}
#     for (key, values_) in emp_values:
#         z_results[key] = {}
#         for eps, vals in values_.items():
#             values = np.array(vals)

#             if len(values.shape) > 1:
#                 values = np.array([item for sublist in values for item in sublist])

#             if all(v == 0 for v in values):
#                 values = np.array([v + np.random.uniform(1e-10, 1e-7) for v in values])
#             # values has no shape bc it is list 
#             values_shape = values.shape
#             values = values.reshape(values_shape[0], -1)

#             min_ = np.array(values).min()
#             max_ = np.array(values).max()

#             x = np.array(values)
#             n = x.shape[0]
#             kde = KDE.KernelDensityEstimator()
#             kde.fit(x)
#             # gss = kde.gss(l_bound=min_.item(), u_bound=max_.item(), fun=kde.score_leave_one_out, tol=1e-5)
#             gss2 = kde.gsection(f=kde.score_leave_one_out, a=min_.item(), b=max_.item(), tol=1e-5)
#             # kde.predict()
#             xpdf = np.linspace(min_.item() - kde.bandwidth * 3, max_.item() + kde.bandwidth * 3, n) #min(n, 500))
#             kde.set_samples(xpdf)
#             score = kde.score_samples(xpdf)
#             est_mise = kde.est_mise(kde.laplacian(), xpdf)

#             z_results[key][eps] = est_mise
#     return z_results


def calc_kde(emp_values):
    z_results = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for (key, emp_values_) in tqdm(emp_values, "Calculating KDE"):
            futures[key] = []
            def task(values_):
                est_mises = {}
                for eps, vals in values_.items(): 0.1,
                    values = np.array(vals)

                    if len(values.shape) > 1:
                        values = np.array([item for sublist in values for item in sublist])

                    if all(v == 0 for v in values):
                        values = np.array([v + np.random.uniform(1e-10, 1e-7) for v in values])

                    # values has no shape bc it is list 
                    values_shape = values.shape
                    values = values.reshape(values_shape[0], -1)

                    min_ = np.array(values).min()
                    max_ = np.array(values).max()

                    x = np.array(values)
                    n = x.shape[0]
                    kde = KDE.KernelDensityEstimator()
                    kde.fit(x)
                    # gss = kde.gss(l_bound=min_.item(), u_bound=max_.item(), fun=kde.score_leave_one_out, tol=1e-5)
                    gss2 = kde.gsection(f=kde.score_leave_one_out, a=min_.item(), b=max_.item(), tol=1e-5)
                    # kde.predict()
                    xpdf = np.linspace(min_.item() - kde.bandwidth * 3, max_.item() + kde.bandwidth * 3, n) #min(n, 500))
                    kde.set_samples(xpdf)
                    score = kde.score_samples(xpdf)
                    est_mise = kde.est_mise(kde.laplacian(), xpdf)
                    est_mises[eps] = est_mise
                return est_mises
            
            try:
                z_results[key] = task(emp_values_)
            except Exception as exc:
                print(exc)

            futures[key] = task(emp_values_)
            # futures[key] = executor.submit(task, emp_values_)

    # for k, future in futures.items():
    #     # z_results[k] = future.result()
    #     # concurrent.futures.as_completed(future)
    #     try:
    #         if future.exception() is not None:
    #             raise future.exception()
    #         else:
    #             z_results[k] = future.result()
    #     except Exception as exc:
    #         print(exc)

    return z_results


def calculate_logit(emp_logit): #, fmodel, dataset, clipped, targets, epsilons):
    # z_results = {}
    # approx = {}
    # calc_diff = {}
    # calc = {}
    # for key, eps in enumerate(epsilons):
    #     approx[eps] = []
    #     calc_diff[eps] = []
    #     calc[eps] = []
    #     for i in range(len(clipped)):
    #         y = targets[i]
    #         y_pred = fmodel(dataset[i])
    #         y_pred_adv = fmodel(clipped[i][key])

    #         logit_benign = emperical._get_logits_diff(y_true=y, y_pred=y_pred) # Benign
    #         logit_attacked = emperical._get_logits_diff(y_true=y, y_pred=y_pred_adv) # Attacked
    #         logit_approx = emperical._get_logits_diff(y_true=torch.argmax(y_pred, axis=1), y_pred=y_pred_adv) # Benign / attacked

    #         logit_diff = logit_attacked / logit_benign
    #         logit_benign = logit_benign.cpu().numpy()
    #         logit_attacked = logit_attacked.cpu().numpy()
    #         logit_approx = logit_approx.cpu().numpy()

    #         logit_diff = logit_diff.cpu().numpy()
    #         logit_diff[np.isnan(logit_diff)] = 0.
    #         logit_diff[logit_diff == -np.inf] = 0.
    #         logit_diff[logit_diff == np.inf] = 0.
            
    #         approx[eps].append(logit_approx)
    #         calc_diff[eps].append(logit_diff)
    #         calc[eps].append(logit_attacked)

    # emp_results = [("approx", approx), ("calc_diff", calc_diff), ("calc", calc)]
    # _z_results = calc_kde(emp_results)
    z_results = calc_kde(emp_logit)
    return z_results


def calculate_f1(emp_f1): #, fmodel, dataset, clipped, targets, epsilons):
    # avg_accuracy = "micro"
    # z_results = {}
    # approx = {}
    # calc_diff = {}
    # calc = {}
    # for key, eps in enumerate(epsilons):
    #     approx[eps] = []
    #     calc_diff[eps] = []
    #     calc[eps] = []
    #     for i in range(len(clipped)):
    #         y = targets[i]
    #         y_pred = torch.argmax(fmodel(dataset[i]), axis=1)
    #         y_pred_adv = torch.argmax(fmodel(clipped[i][key]), axis=1)

    #         y = y.cpu().numpy()
    #         y_pred = y_pred.cpu().numpy()
    #         y_pred_adv = y_pred_adv.cpu().numpy()

    #         f1_benign = metrics.f1_score(y_true=y, y_pred=y_pred, average=avg_accuracy)
    #         f1_attacked = metrics.f1_score(y_true=y, y_pred=y_pred_adv, average=avg_accuracy)
    #         f1_approx = metrics.f1_score(y_true=y_pred, y_pred=y_pred_adv, average=avg_accuracy)
    #         f1_diff = f1_attacked / f1_benign

    #         approx[eps].append(f1_approx)
    #         calc_diff[eps].append(f1_diff)
    #         calc[eps].append(f1_attacked)

    # emp_results = [("approx", approx), ("calc_diff", calc_diff), ("calc", calc)]
    # _z_results = calc_kde(emp_results)
    z_results = calc_kde(emp_f1)
    return z_results


# def calculate_f1(fmodel, dataset, clipped, targets, epsilons):
#     avg_accuracy = "micro"
#     approx = []
#     diff = []
#     calc = []
#     z_results = {"z": {}, "z_real": {}, "y": {}}
#     for key, eps in enumerate(epsilons):
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

#             approx.append(f1_approx)
#             diff.append(f1_attacked / f1_benign)
#             calc.append(f1_attacked)

#         values = np.array(diff)

#         values_shape = values.shape
#         values = values.reshape(values_shape[0], -1)

#         min_ = np.array(values).min()
#         max_ = np.array(values).max()

#         x = np.array(values)
#         n = x.shape[0]

#         kde = KDE.KernelDensityEstimator()

#         kde.fit(x)
#         # gss = kde.gss(l_bound=min_.item(), u_bound=max_.item(), fun=kde.score_leave_one_out, tol=1e-5)
#         gss2 = kde.gsection(f=kde.score_leave_one_out, a=min_.item(), b=max_.item(), tol=1e-5)
#         # kde.predict()

#         xpdf = np.linspace(min_.item() - kde.bandwidth * 3, max_.item() + kde.bandwidth * 3, n) #min(n, 500))

#         kde.set_samples(xpdf)
#         score = kde.score_samples(xpdf)
#         est_mise = kde.est_mise(kde.laplacian(), xpdf)

#         # print('est_mise', est_mise)
#         # print('est bw', kde.bandwidth)
#         # print('silverman 1.06', 1.06*np.std(x)*n**-.2)
#         # print('silverman 0.90', 0.9*np.std(x)*n**-.2)

#         # y = np.linspace(-5, 5, 100)
#         y = values
#         z = kde.score_samples(y)
#         # z[z > 1] = 1
#         zreal = 1 / np.sqrt(2 * np.pi) * np.exp(-y ** 2 / 2)
#         # print(z)
#         # print(zreal)
#         # zreal = np.logical_and(0 < y, y < 1)

#         z_results["y"][eps] = y
#         z_results["z_real"][eps] = zreal
#         z_results["z"][eps] = z

#         plt.plot(y, zreal, label="real")
#         plt.plot(y, z, label="Approx")
#         plt.scatter(y, zreal, label="Real samples")
#         plt.scatter(y, z, label="Approx samples")
#         # plt.plot(values, stats.norm.pdf(values), label="PDF samples")
#         plt.title('PCA')
#         plt.grid()
#         plt.legend()
#         plt.savefig("test123.png")
#         plt.show()

#     return z_results