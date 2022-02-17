import math
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial import distance
from scipy.stats import norm as univariate_normal
from numpy.linalg import norm as L2
DEFAULT_BATCH_SIZE = 20

class KernelDensityEstimator:
    def __init__(self, kernel="multivariate_gaussian", bandwidth_estimator="silverman", univariate_bandwidth=None):
        self.n = 0
        self.const_scorw = 0
        self.d = 0
        self.muk = 0
        self.invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
        self.invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2
        self.min_dists = np.array([])

        kernels = {"multivariate_gaussian": self.kernel_multivariate_gaussian,
                   "univariate_gaussian": self.kernel_univariate_gaussian}
        bandwidth_estimators = {"silverman": self.est_bandwidth_silverman,
                                "scott": self.est_bandwidth_scott,
                                "identity": self.est_bandwidth_identity}
        compatible_estimators = {"multivariate_gaussian": ["silverman", "scott", "identity"],
                                 "univariate": []}

        self.kernel = kernels[kernel]

        # if multivariate gaussian kernel is chosen, choose an estimator
        if kernel == "multivariate_gaussian":
            self.bandwidth_estimator = bandwidth_estimators[bandwidth_estimator]

        # if choosing univariate kernel without bandwidth clarified, print out a warning
        elif kernel == "univariate_gaussian" and (not univariate_bandwidth):
            print("Please define your \"univariate_bandwidth\" parameters since the bandwidth cannot \
                    automatically estimated using univariate kernel yet")

        else:
            self.univariate_bandwidth = univariate_bandwidth

        # Kernel choice
        self.kernel = kernels[kernel]

        # Bandwidth for estimating density
        self.bandwidth = None

        # Store data
        self.data = None

    def kernel_multivariate_gaussian(self, x):
        # Estimate density using multivariate gaussian kernel

        # Retrieve data
        data = self.data

        # Get dim of data
        d = data.shape[1]

        # Estimate bandwidth
        H = self.bandwidth_estimator()
        self.bandwidth = H

        # Calculate determinant of non zeros entry
        diag_H = np.diagonal(H).copy()
        diag_H[diag_H == 0] = 1
        det_H = np.prod(diag_H)

        # Multivariate normal density estimate of x
        var = multivariate_normal(mean=np.zeros(d), cov=H, allow_singular=True)
        density = np.expand_dims(var.pdf(x), 1)
        return density

    def kernel_univariate_gaussian(self, x):
        # Estimate density using univariate gaussian kernel

        # Retrieve data
        data = self.data

        # Get dim of data
        d = data.shape[1]

        # Estimate bandwidth
        h = self.univariate_bandwidth
        # Calculate density
        density = univariate_normal.pdf(L2(x, axis=1)/h)/h

        return density

    def fit(self, X, y=None):
        if len(X.shape) == 1:
            self.data = X[:, np.newaxis]
        else:
            self.data = X

        self.n = len(self.data)

        self.muk = 1 / (2**self.d * np.sqrt(np.pi**self.d))\

        self.const_score = (-self.n * self.d / 2 *
                            np.log(2 * np.pi) - self.n * np.log(self.n - 1))

        self.d = self.data.shape[1]

        return self

    def set_samples(self, data, diff=False):
        if len(data.shape) == 1:
            data = data[:, np.newaxis]
        self.newshape = data.shape[:-1]
        if len(data.shape) == 2:
            self.data_score_samples = data.copy()
        if not len(data.shape) == 2:
            self.data_score_samples = data.reshape(
                (np.prod(self.newshape), data.shape[-1]))

        self.data_dist = distance.cdist(self.data,
                                        self.data_score_samples,
                                        metric='sqeuclidean')

        if diff:
            self.difference = \
                np.zeros((len(self.data),
                          len(self.data_score_samples),
                          self.d))
            for i, datam in enumerate(self.data_score_samples):
                self.difference[:, i, :] = self.data - datam

    def score_samples(self, data = None):
        if data is None:
            # The data is already set. We can compute the scores directly using _logscore_samples
            scores = np.exp(self._logscore_samples())

            # The data needs to be converted to the original input shape
            return scores.reshape(self.newshape)

        # If the input x is a 1D array, it is assumed that each entry corresponds to a
        # datapoint
        # This might result in an error if x is meant to be a single (multi-dimensional)
        # datapoint
        if len(data.shape) == 1:
            data = data[:, np.newaxis]
        if len(data.shape) == 2:
            return np.exp(self._logscore_samples(data))

        # It is assumed that the last dimension corresponds to the dimension of the data
        # (i.e., a single datapoint)
        # Data is transformed to a 2d-array which can be used by self.kde. Afterwards,
        # data is converted to input shape
        newshape = data.shape[:-1]
        scores = np.exp(self._logscore_samples(
            data.reshape((np.prod(newshape), data.shape[-1]))))
        return scores.reshape(newshape)

    def _logscore_samples(self, data = None):
        if data is None:
            eucl_dist = self.data_dist[:self.n]
        else:
            eucl_dist = distance.cdist(
                self.data[:self.n], data, metric='sqeuclidean')

        sum_kernel = np.zeros(eucl_dist.shape[1])
        for dimension in eucl_dist:
            sum_kernel += np.exp(-dimension / (2 * self.bandwidth ** 2))
        const = -self.d/2*np.log(2*np.pi) - np.log(self.n) - \
            self.d*np.log(self.bandwidth)
        return const + np.log(sum_kernel)

    def eval(self, X, y, batch_size=DEFAULT_BATCH_SIZE):
        # Print out evaluation using MSE and CE
        MSE, CE = self.MSE_CE(X, y, batch_size=batch_size)
        print("Cross entropy", CE)
        print("Mean Square Error: ", MSE)
        return MSE, CE

    def MSE_CE(self, X, y, batch_size=DEFAULT_BATCH_SIZE):
        # Calculate mean square error and a binary cross entropy for a given H

        # Retrieve number of classes
        num_classes = len(np.unique(y))

        # Retrieve number of instances in X
        N = len(X)

        # Predict proba
        proba = self.predict_proba(
            X, batch_size=batch_size) + 1e-15  # to fix log(0)

        # Construct mean square error
        MSE = (proba.mean() - 1/num_classes)**2

        # Construct mean cross entropy
        CE = 1/N*np.sum(1/num_classes*np.log(proba) -
                        (1-1/num_classes)*np.log(proba))

        return MSE, CE

    def est_mise(self, laplacian, pdf):
        integral_laplacian = np.trapz(laplacian ** 2, pdf)

        mise_est = 1 / (2 * np.sqrt(np.pi)) / (self.n * self.bandwidth) + \
            integral_laplacian * self.bandwidth ** 4 / 4

        return mise_est

    def est_bandwidth_scott(self):
        # Estimate bandwidth using scott's rule

        # Retrieve data
        data = self.data

        # Get number of samples
        n = data.shape[0]

        # Get dim of data
        d = data.shape[1]

        # Compute standard along each i-th variable
        std = np.std(data, axis=0)

        # Construct the H diagonal bandwidth matrix with std along the diag
        H = (n**(-1/(d+4))*np.diag(std))**2

        return H

    def est_bandwidth_identity(self):
        # Generate an identity matrix of density for bandwidth

        # Retrieve data
        data = self.data

        # Get number of samples
        n = data.shape[0]

        # Get dim of data
        d = data.shape[1]

        # Construct the H bandwidth matrix
        H = np.identity(d)
        return H

    def est_bandwidth_silverman(self):
        # Estimate bandwidth using silverman's rule of thumbs

        # Retrieve data
        data = self.data

        # Get number of samples
        n = data.shape[0]

        # Get dim of data
        d = data.shape[1]

        # Compute standard along each i-th variable
        std = np.std(data, axis=0)

        # Construct the H diagonal bandwidth matrix with std along the diag
        H = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.diag(std)
        return H

    def gss(self, fun, l_bound, u_bound, tol=1e-5, max_n=100):
        """Golden-section search.

        Given a function f with a single local minimum in
        the interval [a,b], gss returns a subset interval
        [c,d] that contains the minimum with d-c <= tol.

        Example:
        >>> f = lambda x: (x-2)**2
        >>> a = 1
        >>> b = 5
        >>> tol = 1e-5
        >>> (c,d) = gss(f, a, b, tol)
        >>> print(c, d)
        1.9999959837979107 2.0000050911830893
        """

        (l_bound, u_bound) = (min(l_bound, u_bound), max(l_bound, u_bound))
        h = u_bound - l_bound
        if h <= tol:
            return (l_bound, u_bound)

        # Required steps to achieve tolerance
        n = int(math.ceil(math.log(tol / h) / math.log(self.invphi)))
        n = max(1, min(n, max_n))

        c = l_bound + self.invphi2 * h
        d = l_bound + self.invphi * h
        yc = fun(c)
        yd = fun(d)

        for k in range(n):
            if yc > yd:
                u_bound = d
                d = c
                yd = yc
                h = self.invphi * h
                c = l_bound + self.invphi2 * h
                yc = fun(c)
            else:
                l_bound = c
                c = d
                yc = yd
                h = self.invphi * h
                d = l_bound + self.invphi * h
                yd = fun(d)

        if yc < yd:
            self.bandwidth = (l_bound + d) / 2
            return (l_bound, d)
        else:
            self.bandwidth = (c + u_bound) / 2
            return (c, u_bound)

    def gsection(self, f, a, b, tol):        
        # Evaluate function at upper and lower bound
        fa = f(a)
        fb = f(b)
        # Compute two new points which correspond to golden ratio
        width = b - a
        
        c = a + self.invphi2*width
        #c = b - self.invphi*width
        #c = a + self.invphi**2*width
        d = a + self.invphi*width
        fc = f(c)
        fd = f(d)
        
        while (b - a) > tol:     
            if fc < fd:
                b = d 
                d = c
                fd = fc
                
                width = self.invphi*width
                c = b - self.invphi*width
                #c = a + self.invphi2*width
                fc = f(c)
            
            else:     
                a = c
                fa = fc
                c = d
                fc = fd
                
                width = self.invphi*width
                d = a + self.invphi*width
                fd = f(d)

        if fc < fd:
            self.bandwidth = (a + d) / 2
            return (a, d)
        else:
            self.bandwidth = (c + b) / 2
            return (c, b)


    def score_leave_one_out(self, bandwidth):

        if self.min_dists.size == 0:
            # print("score min_dists", self.min_dists)
            self.min_dists = distance.squareform(
                distance.pdist(self.data, metric='sqeuclidean')) / 2
            self.min_dists *= -1  # Do it this way to prevent invalid warning

        # Compute the one-leave-out score
        bandwidth = self.bandwidth if bandwidth is None else bandwidth
        score = (np.sum(np.log(np.sum(np.exp(self.min_dists
                                             [:self.n, :self.n] /
                                             bandwidth ** 2),
                                      axis=0) - 1)) -
                 self.n * self.d * np.log(bandwidth) + self.const_score)

        return score

    def predict_proba(self, X, batch_size=10):
        # Predict proba for an input matrix X

        kernel_func = self.kernel

        # Retrieve data
        data = self.data

        # number of samples in data
        n_data = data.shape[0]
        # number of samples in input set
        n_X = X.shape[0]

        # Init the estimated probabilities list
        est_probs = np.empty(0)

        num_batches = np.ceil(n_X/batch_size)
        print("bs:", batch_size)
        for X_ in tqdm(np.array_split(X, num_batches)):
            print("...")

            # Add third dimension for broardcasting
            # shape (1, dim, n_X)
            X_ = np.expand_dims(X, 0).transpose((0, 2, 1))

            # shape(n_data, dim, 1)
            data_ = np.expand_dims(data, 2)

            # The difference of input set and data set pairwise (using broadcasting)
            print(type(X_), type(data_))
            print(X_.shape, data_.shape)
            # shape (n_data, dim, n_X)
            delta = X_ - data_
            print("hier")
            # Flatten the delta into matrix
            delta = delta.reshape(n_data*n_X, -1)  # shape (n_data*n_X, dim)

            est_prob = kernel_func(delta)  # (n_data*n_X, )

            # Calculate mean sum of probability for each sample
            est_prob = 1/n_data*est_prob.reshape(n_data, n_X).T.sum(axis=1)
            est_probs = np.concatenate((est_probs, est_prob))

        return est_probs

    def laplacian(self, data: np.ndarray = None):
        if data is None:
            laplacian = self._laplacian()
            return laplacian.reshape(self.newshape)

        if len(data.shape) == 1:
            data = data[:, np.newaxis]

        if len(data.shape) == 2:
            return self._laplacian(data)

        newshape = data.shape[:-1]
        laplacian = self._laplacian(data.reshape(
            (np.prod(newshape), data.shape[-1])))
        return laplacian.reshape(newshape)

    def _laplacian(self, data: np.ndarray = None):
        if data is None:
            eucl_dist = self.data_dist[:self.n]
        else:
            eucl_dist = distance.cdist(
                self.data[:self.n], data, metric='sqeuclidean')

        laplacian = np.zeros(eucl_dist.shape[1])

        for dimension in eucl_dist:
            pdf = np.exp(-dimension / (2 * self.bandwidth ** 2)) / \
                ((2 * np.pi) ** (self.d / 2) * self.bandwidth ** self.d)

            laplacian += pdf * (dimension / self.bandwidth ** 4 - self.d /
                                self.bandwidth ** 2)

        return laplacian / self.n

    def random_sample(self, scaling_factor):
        # Get H
        H = self.bandwidth_estimator()*scaling_factor

        # Retrieve data
        data = self.data

        # Randomly pick a data point
        random_data = np.random.permutation(self.data)[0]

        # sample
        sample = np.random.multivariate_normal(mean=random_data, cov=H)

        # Print out predicted density for new sample
        print("Density new sample: ", self.predict_proba(
            np.expand_dims(sample, 0))[0])

        return random_data, sample

    def predict(self, X, batch_size=DEFAULT_BATCH_SIZE):
        # Predict proba for a given X to belong to a dataset

        # if x is a vector (has 1 axis)
        if len(X.shape) == 1:
            # expand one more axis to represent a matrix
            X = np.expand_dims(X, 0)

        proba = self.predict_proba(X, batch_size=batch_size)

        return proba
