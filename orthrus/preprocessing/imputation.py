'''
This module defines various classes to impute the dataset. Class instance
must contain the method fit_transform. The output of normalizer.fit_transform(:py:attr:`DataSet.data`)
must have the same number of columns as :py:attr:`DataSet.data`
'''

from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_array
from dataclasses import dataclass

class HalfMinimum(SimpleImputer):
    """
    Performs half-minimum imputation, i.e., finds the non-zero minimum value ``v`` in the data
    and replaces the defined :py:attr:`HalfMinimum.missing_value` with half of ``v``. See
    `SimpleImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer>`_
    for aceptable keyword arguments, this class will ignore :py:attr:`fill_value` and :py:attr:`strategy` keyword
    arguments, as these are calculated internally.
    """

    def __init__(self, **kwargs):
        kwargs['strategy'] = 'constant'  # ensure proper stratgey
        super.__init__(**kwargs)

    def _compute_half_min(self, X, y=None):
        """
        Takes an array and computes half the non-zero minimum value. This value is stored into
        :py:attr:`HalfMinimum.fill_value`.

        Args:
            X (array-like of shape (n_samples, n_features)):

        Returns:
            Inplace method.
        """

        # compute half-minimum
        v = .5*check_array(X)
        v = v[v>0].nanmin()
        self.fill_value = v

    def fit(self, X, y=None):

        # compute fill value
        self._compute_half_min(X, y)

        # call super
        return super.fit(X, y)


