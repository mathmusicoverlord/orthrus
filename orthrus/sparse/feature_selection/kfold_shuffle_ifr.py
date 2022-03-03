"""
This module contains code for implementing an iterative feature removal (IFR) algorithm using leave one
subject out (LOSO) partitions.
"""

# imports
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score as bsr
from orthrus.model_selection.partitioning import TrainPartitioner
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Union, Optional

class KFSIFR(BaseEstimator):
    """
    This iterative feature removal (IFR) algorithm produces ranked feature sets from a single classifier and a data set.

    Parameters:
        classifier (object): The classifier used to select features. The classifier should produce feature weights and ideally
            be sparse, so that relatively few features are weighted heavily compared to the total.

        weights_handle (str): The name of the :py:attr:`KFSIFR.classifier` attribute where the weights are stored. The
            weights stored there should be an ndarray.

        n_splits_kfold (int): The number of folds, ``k`` used in the k-fold cross validation.

        random_state_kfold (int): The random seed used in generating the k-fold partitions. Good for reproducibility
            of results. The default is None.

        train_test_splits (list): An alternate list of (training, test) splits that replace the k-fold cross validation used
            in the algorithm. This is useful if you have a specific set of splits you want to use. The default is None,
            but if provided the arguments :py:attr:`KFSIFR.n_splits_kfold` and :py:attr:`KFSIFR.random_state_kfold` are
            ignored.

        gamma (float): Classification rate used to break the IFR loop. The IFR loop will stop once the number of
            features extracted reaches this proportion of the total features.

        max_feature_threshold (int): The maximum number of features that may be removed on an iteration of IFR.
            Default is None.

        n_splits_shuffle (int): The number of shuffle splits to use for the inner loop. Default is 100.

        random_state_shuffle (int): Random seed for shuffle splits.

        n_top_features (int): The number of top features to remove at each iteration of IFR. If this parameter is not
            given then :py:attr:`KFSIFR.jump_ratio` will be used instead. The default is None.

        jump_ratio (float): The weight ratio used to determine the number of top features to remove for each iteration
            of IFR. For example let :math:`r_i = w_{i}/w_{i+1}` denote the ratio of the
            :math:`i` th largest weight and :math:`i+1` largest weight, then :py:attr:`KFSIFR.jump_ratio` = 2 means
            that top features will be chosen until :math:`r_i \geq 2`. If this parameter is not given then the user
            must provide a constant number of features to prune at each step via the parameter :py:attr:`KFSIFR.n_top_features`.

        sort_freq_classes (bool): If ``True`` the algorithm will sort frequency classes of feature by the mean of
            normalized weight across a LOSO experiment— providing a unique ranking. The default is ``False``.

        imputer (object): Optional imputer to impute training set values with.

    Attributes:
        results_ (DataFrame): Outputs the feature frequencies and rankings for each fold provided by the
            k-fold partition defined by :py:attr:`KFSIFR.n_splits_kfold` and :py:attr:`KFSIFR.random_state_kfold`, or
            the user defined splits defined by :py:attr:`KFSIFR.train_test_splits`.
    """

    def __init__(self,
                 classifier: object,
                 weights_handle: str,
                 n_splits_kfold: int,
                 random_state_kfold=None,
                 train_test_splits=None,
                 gamma: float=.6,
                 max_feature_threshold=None,
                 n_splits_shuffle=100,
                 random_state_shuffle= None,
                 train_prop_shuffle=.8,
                 n_top_features: int=None,
                 jump_ratio: float=None,
                 sort_freq_classes=False,
                 imputer=None):

        # set parameters
        self.classifier = classifier
        if n_splits_kfold != 1:
            self.kfold = StratifiedKFold(n_splits=n_splits_kfold, shuffle=True, random_state=random_state_kfold)
        else:
            self.kfold = TrainPartitioner()
        self.weights_handle = weights_handle
        self.gamma = gamma
        self.max_feature_threshold = max_feature_threshold
        self.n_top_features = n_top_features
        self.jump_ratio = jump_ratio
        self.train_test_splits = train_test_splits
        self.n_splits_shuffle = n_splits_shuffle
        self.n_splits_kfold = n_splits_kfold
        self.random_state_kfold = random_state_kfold
        self.random_state_shuffle = random_state_shuffle
        self.train_prop_shuffle = train_prop_shuffle
        self.sort_freq_classes = sort_freq_classes
        self.imputer = imputer

        # set attributes
        self.results_ = pd.DataFrame()

    def fit(self, X, y, groups=None):
        """
        Performs the feature selection algorithm and stores the results in the attribute :py:attr:`KFSIFR.results_`.

        Args:
            X (array-like of shape (n_samples, n_features)): The data array used to select features via :py:attr:`KFSIFR.classifier`.

            y (array-like of shape (n_samples, )): The labels used to select features via :py:attr:`KFSIFR.classifier`.

            groups (array-like of shape (n_samples, )): Optional set up of labels defining the groups used in a
             Leave-One-Group-Out experiment. This is useful if you don't want members of the same group in both the
             training in test for a LOSO fold.

        Returns:
            inplace method.

        """
        # check array
        X, y = check_X_y(X, y)

        # set total feature set
        St = np.arange(X.shape[1])

        # intialize results
        self.results_ = pd.DataFrame(index=St)

        # set default value of maximum feature threshold
        if self.max_feature_threshold is None:
            max_feature_threshold = X.shape[1]
        else:
            max_feature_threshold = self.max_feature_threshold

        # loop through cv folds
        if self.train_test_splits is None:
            splits = self.kfold.split(X, y)
        else:
            splits = self.train_test_splits
        for i, (kf_train_index, kf_test_index) in enumerate(splits):
            print("Starting fold " + str(i) + "...")
            if self.imputer is None:
                Xi_train = X[kf_train_index, :]
            else:
                print("Imputing...")
                Xi_train = self.imputer.fit_transform(X[kf_train_index, :])

            yi_train = y[kf_train_index]

            # intialize feature set
            Si = pd.DataFrame(index=St, columns=np.arange(Xi_train.shape[0])).fillna(False).astype(bool)

            if groups is None:
                groups_train = np.array(kf_train_index)
                shuffle_splits = ShuffleSplit(n_splits=self.n_splits_shuffle,
                                              random_state=self.random_state_shuffle,
                                              train_size=self.train_prop_shuffle).split(Xi_train, yi_train)
            else:
                groups_train = np.array(groups)[kf_train_index]
                shuffle_splits = GroupShuffleSplit(n_splits=self.n_splits_shuffle,
                                                   random_state=self.random_state_shuffle,
                                                   train_size=self.train_prop_shuffle).split(Xi_train, yi_train, groups_train)

            for j, (shuffle_train_index, shuffle_test_index) in enumerate(shuffle_splits):
                print("Starting shuffle split " + str(j) + "...")
                Xij_train = Xi_train[shuffle_train_index, :]; yij_train = yi_train[shuffle_train_index]
                Xij_valid = Xi_train[shuffle_test_index, :]; yij_valid = yi_train[shuffle_test_index]
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
                            id = np.where(f_ratios > self.jump_ratio)[0][0]
                            if (id + 1) < max_feature_threshold:
                                S = S[:(id+1)]
                                S = Sc[S]
                            else:
                                print("Maximum feature threshold exceeded, no features selected...")
                                break
                        except IndexError:
                            print("Jump failed, no features selected...")
                            break
                    else:
                        S = Sc[S[:self.n_top_features]]

                    # record accuracy on validation set
                    self.classifier.fit(Xij_train[:, S], yij_train)
                    yij_pred = self.classifier.predict(Xij_valid[:, S])
                    score = bsr(yij_valid, yij_pred)
                    print(r"BSR = %.2f%%" % (score*100,))

                    Sij = np.array(list(set(Sij).union(set(S))))
                    print("Feature set size = " + str(Sij.size))

                    # check threshold condition
                    if score < self.gamma:
                        break

                # set selected features
                Si.loc[Sij, j] = True

            # compute frequencies and sort
            Si = Si.sum(axis=1)
            if self.sort_freq_classes:
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

                # rank features
                Si = Si.to_frame().rename({0: 'Frequency_' + str(i)}, axis='columns')
                if self.sort_freq_classes:
                    Si['Rank_' + str(i)] = pd.NA
                    Si.loc[index_list, 'Rank_' + str(i)] = np.arange(len(index_list))

            # store results
            self.results_ = pd.concat([self.results_, Si], axis=1)

            return self
        
    def transform(self,
                  X,
                  fold: Union[int, str] = 'mean',
                  n_top_features: Optional[Union[int, float]] = None):
        """
        Restricts :py:attr:`X` to the top ranked features. If :py:attr:`KFSIFR.sort_freq_classes` is ``False`` then
        the transform method will rank off of the frequencies, otherwise it will use the rankings available.

        Args:
            X (array-like of shape (n_samples, n_features)): The data to restrict features on.
            fold (int or str): If an integer then the ranking of the feature will be decided on specfied fold.
                One can also specify :py:attr:`fold` = "mean", indicating to use the mean rank across folds.

            n_top_features (int, float, None): Specifies the number of top features to restrict to, if no value
                is given the transform method will restrict to the top 1% of the features. If a float between 0
                and 1 is provided the :py:attr:`n_top_features` will be computed by the given proportion of the features.

        Returns:
            (array-like of shape (n_samples, n_top_features)) : Restrict array to the top features.

        """
        # check for fit
        check_is_fitted(self)

        # grab results
        if self.sort_freq_classes:
            rank_col = "Rank"
        else:
            rank_col = "Frequency"

        # compute ranks
        ranks = self.results_.filter(regex=rank_col+'*')

        # grab relevant rankings
        if fold == "mean":
            ranks = ranks.mean(axis=1)
        else:
            ranks = ranks['%s_%s' % (rank_col, str(fold))]

        # convert to sort values
        if rank_col == "Rank":
            ranks = ranks.values
        elif rank_col == "Frequency":
            ranks = (-ranks).argsort().values
        else:
            raise ValueError("Invalid ranking column given!")

        # calculate n_top_features if necessary
        if n_top_features is None:
            n_top_features = np.round(.01*X.shape[1])
        if type(n_top_features) == float:
            assert(n_top_features>0 and n_top_features<=1,
                   "Proportion must be provided as a float between 0 and 1.")
            n_top_features = np.round(n_top_features*X.shape[1])

        # restrict X
        feature_ids = ranks < n_top_features

        return X[:, feature_ids]
