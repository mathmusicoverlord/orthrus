import numpy as np
'''
This module defines various classes to normalize the dataset. Class instance 
must contain the method fit_transform. The output of normalizer.fit_transform(:py:attr:`DataSet.data`) 
must have the same number of columns as :py:attr:`DataSet.data`
'''

class LogNormalizer():
    def fit_transform(self, X):
        """
        Applies element wise log to the data. Any values that are negative infinity are set to zero after applying log.

        Args:
            X (ndarray of shape (m, n))): array of data, with m the number of observations in R^n.
        Return:
            (ndarray of shape (m, n))): Modified data matrix.
        """
        with np.errstate(divide='ignore'):
            data = np.log(X)
            data[np.isneginf(data)] = 0

        return data