import numpy as np
'''
This module defines various classes to normalize the dataset. Class instance 
must contain the method fit_transform. The output of normalizer.fit_transform(:py:attr:`DataSet.data`) 
must have the same number of columns as :py:attr:`DataSet.data`
'''

class LogNormalizer():
    def fit_transform(self, data):
        with np.errstate(divide='ignore'):
            data = np.log(data)
            data[np.isneginf(data)] = 0

        return data