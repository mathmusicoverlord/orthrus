"""
This module contains a class which implements the PCA-SNR feature selection method.
"""

# imports
import torch as tc
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA


class PCASNR(BaseEstimator):

    def __init__(self):

        # attributes
        self.pca_ = PCA(n_components=1, whiten=True)
        self.snr_plot = None
        self.results_ = pd.DataFrame()

    def snr(self, X, y, w):
        # define the classes
        classes = np.unique(y)
        class_0_ids = np.where(y == classes[0])[0]
        class_1_ids = np.where(y == classes[1])[0]

        # compute the class means and covariances
        X0 = X[class_0_ids, :]
        X1 = X[class_1_ids, :]
        mu0 = np.mean(X0, axis=0).reshape(-1, 1)
        mu1 = np.mean(X1, axis=0).reshape(-1, 1)
        mu = mu0 - mu1
        cov0 = np.cov(X0.transpose())
        cov1 = np.cov(X1.transpose())
        cov = cov0 + cov1
        cov = cov.reshape(X.shape[1], X.shape[1])

        # compute the SNR^2
        snr = np.square(np.matmul(w, mu)) / np.matmul(np.matmul(w, cov), w.transpose())

        return snr.item()

    def fit(self, X, y):

        # convert data types
        X = np.array(X)
        y = np.array(y)

        # check for two classes
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("Method currently only accepts two class problems.")

        # get data dimensions
        m = X.shape[0]
        n = X.shape[1]

        # initialize results
        self.results_ = pd.DataFrame(index=np.arange(n), columns=['SNR'])

        # start the loop
        for i in range(n):

            # restrict the features in the data
            Xi = X[:, :(i+1)]

            # fit the pca model and extract PC 1
            self.pca_.fit(Xi)
            w = self.pca_.components_[0, :].reshape(1, -1)

            # compute the SNR w.r.t. the two classes
            print(r"Computing SNR on top %d features..." % (i+1,))
            snr = self.snr(Xi, y, w)

            # store the snr to the results
            self.results_.at[i, 'SNR'] = snr

        # generate plot
        self.plot_snrs()

        return self

    def plot_snrs(self):

        # plot the data frame
        self.snr_plot = self.results_['SNR'].plot(title='SNR w.r.t PC 1 across top features',
                                                  xlabel='Top Features',
                                                  ylabel='SNR',
                                                  linewidth=2)






