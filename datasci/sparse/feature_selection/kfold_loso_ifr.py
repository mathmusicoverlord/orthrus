"""
This module contains code for implementing an iterative feature removal (IFR) algorithm using leave one
subject out (LOSO) partitions.
"""

# imports
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
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
                 kfold_labels=None):

        # set parameters
        self.scorer = scorer
        self.classifier = classifier
        self.kfold = StratifiedKFold(n_splits=n_splits_kfold, shuffle=True, random_state=random_state_kfold)
        self.weights_handle = weights_handle
        self.gamma = gamma
        self.n_top_features = n_top_features
        self.jump_value = jump_value
        self.kfold_labels = kfold_labels

        # set attributes
        self.loso = LeaveOneOut()
        self.results_ = pd.DataFrame()

    def fit(self, X, y):

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

            for j, (loso_train_index, loso_test_index) in enumerate(self.loso.split(Xi_train, yi_train)):
                print("Starting LOSO split " + str(j) + "...")
                Xij_train = Xi_train[loso_train_index, :]; yij_train = yi_train[loso_train_index]
                Xij_valid = Xi_train[loso_test_index, :]; yij_valid = yi_train[loso_test_index]
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
                        f_ratios = f_weights_sorted[:-1]/f_weights_sorted[1:]
                        f_ratios[np.isinf(f_ratios)] = 0
                        f_ratios[np.isnan(f_ratios)] = 0
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

            # compute frequencies and store into results
            Si = Si.sum(axis=1)
            self.results_[i] = Si