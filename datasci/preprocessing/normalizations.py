import numpy as np
from copy import deepcopy
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

class MedianFoldChangeNormalizer():
    def fit_transform(self, X, controls=None):

        # recast as numpy array is necessary
        Z = deepcopy(np.array(X))

        # set default for controls
        if controls is None:
            controls = np.arange(Z.shape[0])

        # Find missing data and change zero to nan
        missing = (Z == 0)
        Z[missing] = np.nan

        # Integral normalize
        integral = np.nansum(Z, axis=1)
        integral = integral.reshape(-1, 1)
        Z = 100 * (Z / integral)

        # Calculate medians of controls for "Golden Reference"
        median = np.nanmedian(Z[controls, :], axis=0)

        # Calculate fold changes from median
        quotients = Z / median

        # Calculate median of fold changes
        median_quotients = np.nanmedian(quotients, axis=1)
        median_quotients = median_quotients.reshape(-1, 1)

        # Scale each sample by median of the fold changes
        Z = Z / median_quotients
        Z[missing] = 0

        return Z