import sys
import torch
import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn import metrics
import matplotlib.pyplot as plt
import concurrent
from concurrent.futures.thread import ThreadPoolExecutor

sys.path.insert(0, '../')

from project.completeness import KDE
from project.robustness import emperical

def calc_kde(emp_values):
    z_results = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for (key, emp_values_) in tqdm(emp_values, "Calculating KDE"):
            futures[key] = []
            def task(values_):
                est_mises = {}
                for eps, vals in values_.items():
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

    return z_results


def calculate_logit(emp_logit):
    z_results = calc_kde(emp_logit)
    return z_results


def calculate_f1(emp_f1):
    z_results = calc_kde(emp_f1)
    return z_results