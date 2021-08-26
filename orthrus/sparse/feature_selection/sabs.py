# imports
import torch as tc
import numpy as np
from sklearn.base import BaseEstimator

class SABS(BaseEstimator):

    def __init__(self,
                 a=1.0,
                 b=1.0,
                 alpha=.001,
                 iterations=100,
                 store_updates=False,
                 use_cuda=False,
                 device=None):

        # Parameters
        self.a = a
        self.b = b
        self.alpha = alpha
        self.use_cuda = use_cuda
        self.device = device
        self.iterations = iterations
        self.store_updates = store_updates

        # Attributes
        self.var_Y_ = None
        self.var_Z_ = None
        self.cov_YZ_ = None
        self.f_ = None
        self.u_ = None
        self.v_ = None
        self.w_ = None
        self.weights_ = None

    def fit(self, X, y):

        # split by the two classes
        labels = np.unique(y)

        if len(labels) < 1 or len(labels) > 2:
            raise ValueError("This feature selection method currently only works for two class problems.")

        # generate data matrices
        y_Y = (np.array(y) == labels[0])
        y_Z = (np.array(y) == labels[1])
        Y = X[y_Y, :]
        Z = X[y_Z, :]

        # get a PCA basis for Y and Z resp.
        svd = np.linalg.svd
        Y = svd(Y, full_matrices=False)[2]
        Z = svd(Z, full_matrices=False)[2]

        # grab dimensions of data
        m = Y.shape[0]
        p = Z.shape[0]
        n = X.shape[1]

        # transfer to torch tensors
        if self.use_cuda:
            if self.device is None:
                dev = tc.device('cuda:0')
            else:
                dev = tc.device('cuda:'+str(self.device))
        else:
            dev = tc.device('cpu')

        # transfer arrays to set device (cpu or gpu)
        Y_d = tc.tensor(Y, device=dev)
        Z_d = tc.tensor(Z, device=dev)

        # calculate dtype
        dtype = Y_d.dtype

        # calculate initial variance
        alpha = Y_d.sum(axis=0).norm()
        beta = Z_d.sum(axis=0).norm()

        # set update parameters
        u = 2 * tc.ones((m, 1), dtype=dtype, device=dev) / alpha # ensures u'*Y*diag(delta)^2*Y'u = 1 initially
        v = 2 * tc.ones((p, 1), dtype=dtype, device=dev) / beta # ensures v'*Z*diag(delta)^2*Z'v = 1 initially
        w = tc.zeros(n, dtype=dtype, device=dev)

        # switch direction for max covariance
        with tc.no_grad():
            h = (tc.tanh(w) + 1) / 2
            u_Y = tc.matmul(u.transpose(0, 1), tc.matmul(Y_d, tc.diag(h)))
            v_Z = tc.matmul(v.transpose(0, 1), tc.matmul(Z_d, tc.diag(h)))
            cov_YZ = tc.matmul(u_Y, v_Z.transpose(0, 1))

            if cov_YZ < 0:
                u = - u

        # initialize update cache
        if self.store_updates:
            self.var_Y_ = np.zeros(self.iterations)
            self.var_Z_ = np.zeros(self.iterations)
            self.cov_YZ_ = np.zeros(self.iterations)
            self.f_ = np.zeros(self.iterations)
            self.u_ = np.zeros((m, self.iterations))
            self.v_ = np.zeros((p, self.iterations))
            self.w_ = np.zeros((n, self.iterations))


        # compute gradients
        for i in range(self.iterations):
            # print update
            print("Starting iteration " + str(i) + "...")

            # set grad true on tensors
            u.requires_grad = True
            v.requires_grad = True
            w.requires_grad = True

            # diagonal feature matrix
            h = (tc.tanh(w) + 1) / 2

            # linear combinations
            u_Y = tc.matmul(u.transpose(0, 1), tc.matmul(Y_d, tc.diag(h)))
            v_Z = tc.matmul(v.transpose(0, 1), tc.matmul(Z_d, tc.diag(h)))

            # compute variance terms
            var_Y = tc.matmul(u_Y, u_Y.transpose(0, 1))
            print("Class A vector variance: " + str(var_Y.item()))
            var_Z = tc.matmul(v_Z, v_Z.transpose(0, 1))
            print("Class B vector variance: " + str(var_Z.item()))
            cov_YZ = tc.matmul(u_Y, v_Z.transpose(0, 1))
            print("Covariance: " + str(cov_YZ.item()))

            # define the objective functions
            f = cov_YZ - \
                self.a*tc.pow(var_Y - 1, 2) - \
                self.a*tc.pow(var_Z - 1, 2) - \
                self.b * tc.norm(h, p=1)

            # calculate back prop and update u, v, and w
            f.backward()
            print("Objective function: " + str(f.item()))
            with tc.no_grad():
                u = u + self.alpha*u.grad # maximize correlation (minimize angle)
                v = v + self.alpha*v.grad # maximize correlation (minimize angle)
                w = w + self.alpha*w.grad # minimize feature space, maximize correlation (maximize angle)

                if self.store_updates:
                    self.var_Y_[i] = var_Y.item()
                    self.var_Z_[i] = var_Z.item()
                    self.cov_YZ_[i] = cov_YZ.item()
                    self.f_[i] = f.item()
                    self.u_[:, i] = u.detach().cpu().numpy().reshape(-1,)
                    self.v_[:, i] = v.detach().cpu().numpy().reshape(-1,)
                    self.w_[:, i] = w.detach().cpu().numpy().reshape(-1,)

            # space output
            print("")
            print("")

        # store final weights
        self.weights_ = ((tc.tanh(w) + 1) / 2).detach().cpu().numpy()




