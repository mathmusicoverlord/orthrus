"""
This module contains code for implementing an iterative feature removal (IFR) algorithm using leave one
subject out (LOSO) partitions.
"""

# imports
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeaveOneOut
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class KFLIFR(BaseEstimator):

    def __init__(self,
                 scorer: object,
                 classifier: object,
                 weights_handle: str,
                 n_splits_kfold: int,
                 random_state_kfold=None,
                 gamma: float = .01,
                 n_top_features: int = None,
                 jump_value: float = None,
                 kfold_labels=None,
                 sort_occurence_classes=False):

        # set parameters
        self.scorer = scorer
        self.classifier = classifier
        self.kfold = StratifiedKFold(n_splits=n_splits_kfold, shuffle=True, random_state=random_state_kfold)
        self.weights_handle = weights_handle
        self.gamma = gamma
        self.n_top_features = n_top_features
        self.jump_value = jump_value
        self.kfold_labels = kfold_labels
        self.sort_occurence_classes=sort_occurence_classes

        # set attributes
        self.results_ = pd.DataFrame()

    def fit(self, X, y, groups=None):

        # set total feature set
        St = np.arange(X.shape[1])

        # intialize results
        self.results_ = pd.DataFrame(index=St, columns=np.arange(self.kfold.n_splits))

        # loop through cv folds
        if self.kfold_labels is None:
            splits = self.kfold.split(X, y)
        else:
            splits = self.kfold_labels
        for i, (kf_train_index, kf_test_index) in enumerate(splits):
            print("Starting fold " + str(i) + "...")
            Xi_train = X[kf_train_index, :]; yi_train = y[kf_train_index]
            Xi_test = X[kf_test_index, :]; yi_test = y[kf_test_index]

            # intialize feature set
            Si = pd.DataFrame(index=St, columns=np.arange(Xi_train.shape[0])).fillna(False).astype(bool)

            if groups is None:
                logo_splits = LeaveOneOut().split(Xi_train, yi_train)
            else:
                groups_train = np.array(groups)[kf_train_index]
                logo_splits = LeaveOneGroupOut().split(Xi_train, yi_train, groups_train)

            for j, (logo_train_index, logo_test_index) in enumerate(logo_splits):
                print("Starting LOGO split " + str(j) + "...")
                Xij_train = Xi_train[logo_train_index, :]; yij_train = yi_train[logo_train_index]
                Xij_valid = Xi_train[logo_test_index, :]; yij_valid = yi_train[logo_test_index]
                Sij = [] # intial feature set
                while True:
                    print("Starting IFR pass...")

                    # calculate feature set complement
                    Sc = np.array(list(set(St).difference(set(Sij))))

                    # fit the classifier
                    self.classifier.fit(Xij_train[:, Sc], yij_train)

                    # select top features
                    f_weights = np.abs(eval("self.classifier" + "." + self.weights_handle))
                    S = np.argsort(-f_weights)
                    if self.n_top_features is None:
                        f_weights_sorted = f_weights[S]
                        a = f_weights_sorted[:-1]
                        b = f_weights_sorted[1:]
                        f_ratios = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
                        try:
                            id = np.where(f_ratios > self.jump_value)[0][0]
                            S = S[:id]
                            S = Sc[S]
                        except IndexError:
                            print("Jump failed, no features selected...")
                            break
                    else:
                        S = Sc[S[:self.n_top_features]]

                    # record accuracy on validation set
                    #self.classifier.fit(Xij_train[:, S], yij_train)
                    #yij_pred = self.classifier.predict(Xij_valid[:, Sc])
                    #score = self.scorer(yij_valid, yij_pred)
                    #print("acc. score = " + str(score))
                    Sij = np.array(list(set(Sij).union(set(S))))
                    print("Feature set size = " + str(Sij.size))
                    # check threshold condition
                    if Sij.size > self.gamma*St.size:
                        break

                # set selected features
                Si.loc[Sij, j] = True

            # compute frequencies and sort
            Si = Si.sum(axis=1)
            if self.sort_occurence_classes:
                print("Sorting each occurrence class.")
                freqs = np.sort(Si.unique())[::-1]
                index_list = []
                for k in freqs:
                    print("Sorting occurrence class " + str(k) + "...")
                    S_freq = np.array(Si[Si == k].index)
                    A = np.zeros((len(S_freq), np.unique(groups_train).size))
                    logo_splits = LeaveOneGroupOut().split(Xi_train, yi_train, groups_train)
                    for j, (logo_train_index, logo_test_index) in enumerate(logo_splits):
                        print("Starting LOGO split " + str(j) + "...")
                        Xij_train = Xi_train[logo_train_index, :]; yij_train = yi_train[logo_train_index]

                        # fit the classifier
                        self.classifier.fit(Xij_train[:, S_freq], yij_train)

                        # store weights
                        f_weights = np.abs(eval("self.classifier" + "." + self.weights_handle))
                        A[:, j] = f_weights.reshape(-1,)

                    # normalize weights and then compute means
                    A = A / A.sum(axis=0)
                    means = A.mean(axis=1)
                    order = (-means).argsort()

                    # sort S_freq and append to list
                    S_freq = S_freq[order]
                    index_list = index_list + S_freq.tolist()

                # sort features
                Si = Si.loc[index_list]

            # store results
            self.results_[i] = Si