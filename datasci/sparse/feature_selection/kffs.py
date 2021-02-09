# imports
import torch as tc
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

class KFFS(BaseEstimator):
    """
    K-fold Feature Selection (kFFS) selects features using a classifier in a k-fold cross-validation experiment.
    Specifically, the given classifier is trained on each fold, the features in each fold are ranked and sorted,
    the top n features are selected from each fold, and the features across all folds are collected and ranked
    by how many times a given feature was in the top n across each fold.

    Parameters:
        k (int): Indicates the number of folds to use in k-fold partition. Default is 5.

        n: (int): Indicates the rank threshold to use for each fold. KFFS grabs the top ``n`` features from each fold.

        classifier (object): Class instance of the classifier being used, must contain a ``fit`` method.

        f_weights_handle (str): Name of the classifier attribute containing the feature weights for a given fold.

        f_rnk_func (object): Function to be applied to feature weights for feature ranking. Default is None, and the
            features will be ranked in from least to greatest.

        random_state (int): Random state to generate k-fold partitions. Default is 0.

    Attributes:
        _classifiers (series): Contains each classifier trained on each fold.
        _ranks (ndarray): Contains the final rankings of all the features.
        _weights (ndarray): Contains the feature weights and ranks for each fold, and the final rankings.

    Examples:

    """

    def __init__(self, k=5, n=10, classifier=None, f_weights_handle=None, f_rnk_func=None, random_state=0):

        # set parameters
        self.k = k
        self.n = n
        self.classifier = classifier
        self.random_state = random_state
        self.f_weights_handle = f_weights_handle
        self.f_rnk_func = f_rnk_func

        # set attributes
        self._classifiers = pd.Series()
        self._ranks = None
        self._weights = None

    def fit(self, X, y, **kwargs):

        # generate splits
        kfold = KFold(n_splits=self.k, shuffle=True, random_state=self.random_state)
        splits = kfold.split(X, y)

        # set returns
        feature_index = np.arange(0, np.shape(X)[1])
        f_weight_results = pd.DataFrame(index=feature_index)
        classifiers = pd.Series()

        # loop over splits
        for i, (train_index, test_index) in enumerate(splits):
            X_train = X[train_index, :]
            y_train = y[train_index]

            # fit classifier
            self.classifier.fit(X_train, y_train)
            self._classifiers['classifier_' + str(i)] = self.classifier

            if not (self.f_weights_handle is None):
                f_weights_name = "weights_" + str(i)
                f_weights = eval("self.classifier" + "." + self.f_weights_handle)
                f_weight_results[f_weights_name] = np.nan
                f_weight_results.loc[feature_index , f_weights_name] = pd.Series(index=feature_index , data=f_weights)
                if not (self.f_rnk_func is None):
                    f_rnk_name = "ranks_" + str(i)
                    weights = f_weight_results.loc[feature_index, f_weights_name]
                    f_weight_results[f_rnk_name] = np.nan
                    f_weight_results.loc[feature_ids, f_rnk_name] = (-self.f_rnk_func(weights)).argsort()
                    f_weight_results[f_rnk_name] = f_weight_results[f_rnk_name].astype('Int64')
                else:
                    f_rnk_name = method_name + "ranks_" + str(i)
                    weights = f_weight_results.loc[feature_index, f_weights_name]
                    f_weight_results[f_rnk_name] = np.nan
                    f_weight_results.loc[feature_index, f_rnk_name] = (-np.array(weights)).argsort()
                    f_weight_results[f_rnk_name] = f_weight_results[f_rnk_name].astype('Int64')

            # set weights
            self._results = f_weight_results

            # perform kffs
            threshold = self.n
            top_features = self._results.filter(regex='ranks', axis=1) < threshold
            occurences = top_features.astype(int).sum(axis=1)
            self._results['top_' + str(threshold) + '_occurences'] = occurences.replace(0, pd.NA)
            self._results['top_' + str(threshold) + '_rank'] = occurences.argsort()
            self._ranks = occurences.argsort().values







