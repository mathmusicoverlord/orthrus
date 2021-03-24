"""
This module contains methods related to general data decompositions outside of the standard sklearn library
"""
import torch as tc
import numpy as np
from sklearn.base import BaseEstimator

class OrthTransform(BaseEstimator):
    """
    This class takes a subspace :py:math:`S` (:py:attr:`OrthTransform.subspace`) and affine shift :py:math:`x_0`
    (:py:attr:`OrthTransform.shift`) and resolves data :py:math:`X`, with :py:math:`m` rows (samples) and :py:math:`n`
    columns (features), onto :py:math:`S` and its orthogonal complement :py:math:`S^{\perp}`. The data resolved onto
    :py:math:`S^{\perp}` is then transformed by a function :py:math:`T`, represented by the ``fit_transform`` method of
    :py:attr:`OrthTransform.transformer`,to reduce the dimension of the orthogonal components.

    Parameters:
        subspace (ndarray of shape (n_features, subspace_dimension)): The matrix whose columns span the subspace in
            consideration.
        shift (1-d array with n_features components): The point in space to affinely shift the data by— we subtract the
            point.
        transformer (class instance): The transformation object to transform the orthogonal complement to the subspace—
            must have a ``fit_transform`` method.
        transformer_args (dict): The dictionary of arguments to be passed to the transformers ``fit`` method. The
            default is an empty dictionary.

    Attributes:
        coordinates (ndarray of shape (n_samples, n_components)): The embedding produced by OrthTransform where
            ``n_components`` is the number of components given in :py:attr:`OrthTransform.subspace` combined with
            the number of components given in :py:attr:`OrthTransform.transformer`.
    """
    def __init__(self, subspace, shift, transformer, transformer_args=dict()):
        # parameters
        self.subspace = subspace
        self.shift = shift
        self.transformer = transformer
        self.transformer_args = transformer_args

        # attributes
        self.coordinates_ = None

    def fit(self, X, y=None):
        """
        Computes the transformed coordinates of the orthogonal complement to the shifted data, and post-concatenates
        this to the coordinates of the shifted data projected into the subspace.

        Args:
            X (array-like of shape (n_samples, n_samples)): An array representing the data.
            y (Ignored):

        Returns:
            OrthTransform: The fit OrthTransform instance.

        """
        # shift data
        X = np.array(X) - np.array(self.shift).reshape(1, -1)

        # project data onto subspace
        P = np.linalg.qr(self.subspace)[0]
        PX = np.matmul(X, P)

        # subtract off projected component
        Q = X - np.matmul(PX, P.transpose())

        # transform the orthogonal component
        TQ = self.transformer.fit_transform(Q, **self.transformer_args)

        # concatenate components
        self.coordinates_ = np.hstack((PX, TQ))

        return self

    def fit_transform(self, X, y=None):
        """
        Fits the orthogonal transform model to the data via :py:meth:`OrthTransform.fit`, returns the embedding stored
        in :py:attr:`OrthTransform.coordinates_`.

        Args:
            X (array-like of shape (n_samples, n_samples)): An array representing the data.
            y (Ignored):

        Returns:
            ndarray of shape (n_samples, n_components): The embedding produced by OrthTransform where ``n_components``
                is the number of components given in :py:attr:`OrthTransform.subspace` combined with the number of
                components given in :py:attr:`OrthTransform.transformer`

        """

        # import
        from copy import deepcopy

        # first fit the model
        self.fit(X, y)

        return deepcopy(self.coordinates_)