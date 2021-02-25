"""
This module contains classes implementing manifold learning algorithms.
"""

# imports
import torch as tc
import numpy as np
from sklearn.base import BaseEstimator

class MDS(BaseEstimator):
    """
    This class compute the Multidimensional Scaling (MDS) embedding of a distance matrix X for ``n_components``. So that
    the data is embedding into Euclidean space with dimension n_components. The algorithm is as follows:
    1. Compute :math:`D = X\odot X` (Distances squared).
    2. Compute :math:`B = CDC` where :math:`C` is the double-centering matrix :math:`C=I - \frac{1}{\text{n\_components}}ee^T`
        and :math:`e` is the ones vector of dimension ``n_components``
    3. Compute :math:`E`, the matrix whose columns are the eigenvectors of :math:`B`,  :math:`\lambda` the vector of
        corresponding eigenvalues.
    4. Let :math:`\tilde{\Lambda}` be the diagonal matrix containing the top ``n_components`` largest eigenvalues in
        descending order, and let :math:`\tilde{E}` be the matrix whose columns are the eigenvectors, columns
        of :math:`E`, that correspond to the diagonal entries in :math:`\tilde{\Lambda}`.
    5. The embedding is given by :math:`Y = \tilde{E}\tilde{\Lambda}^{\frac{1}{2}}`.
    Parameters:
        n_components (int): The number of components (dimensions) to use for the embedding.
        use_cuda (bool): Flag indicating whether or not to use cuda tensors on the gpu.

    Attributes:
        eigenvalues_ (ndarray of shape (n_samples,)): The eigenvalues corresponding to the MDS embedding.
        eigenvectors_ (ndarray of shape (n_samples, n_samples)): The eigenvectors corresponding to the MDS embedding.
        embedding_ (ndarray of shape (n_samples, n_components)): The embedding given by MDS with n_components.
    """

    def __init__(self,
                 n_components=2,
                 use_cuda=False):


        # set parameters
        self.n_components = n_components
        self.use_cuda = use_cuda

        # set attributes
        self.eigenvalues_ = np.array([])
        self.eigenvectors_ = np.array([])
        self.embedding_ = np.array([])

    def fit(self, X, y=None):
        """
        Computes the position of the points in the embedding space. Stores the eigenvalues and eigenvectors into
        :py:attr:`MDS.eigenvalues_` and :py:attr:`MDS.eigenvectors_` respectively. Stores the embedding into
        :py:attr:`MDS.embedding_`

        Args:
            X (array-like of shape (n_samples, n_samples)): An array representing the distances between samples.
                Should be a symmetric non-negative matrix.
            y (Ignored):

        Returns: Inplace method.
        """

        # check if using cuda
        if self.use_cuda:
            device = 'cuda:0'
        else:
            device = 'cpu'

        # transfer X to device
        X = tc.tensor(X, device=device, dtype=tc.float)

        # check dimensions
        m, n = X.shape
        if m != n:
            raise ValueError("Input must be a square symmetric matrix representing distances")

        # compute square distances
        X = tc.pow(X, 2)

        # double center
        B = -(.5)*(X - tc.mean(X, 0).reshape(1, -1))
        B = B - tc.mean(B, 1).reshape(-1, 1)

        # compute eigenvalue decomposition
        eigenvalues, eigenvectors = tc.eig(B, eigenvectors=True)
        eigenvalues = eigenvalues[:, 0]

        # store eigen-decomposition
        self.eigenvalues_ = eigenvalues = eigenvalues.cpu().numpy()
        self.eigenvectors_ = eigenvectors = eigenvectors.cpu().numpy()

        # compute embedding
        n_components = self.n_components
        idx = ((-eigenvalues).argsort())[:n_components]
        embedding = eigenvectors[:, idx] * np.sqrt(eigenvalues[idx]).reshape(1, -1)

        # store embedding
        self.embedding_ = embedding

    def fit_transform(self, X, y=None):
        """
        Fits the MDS model to the data via :py:meth:`MDS.fit`, returns the embedding stored in :py:attr:`MDS.embedding_`.

        Args:
            X (array-like of shape (n_samples, n_samples)): An array representing the distances between samples.
                Should be a symmetric non-negative matrix.
            y (Ignored):

        Returns:
            ndarray of shape (n_samples, n_features): The embedding produced my MDS with n_components

        """
        # import
        from copy import deepcopy

        # first fit the model
        self.fit(X, y)

        return deepcopy(self.embedding_)











