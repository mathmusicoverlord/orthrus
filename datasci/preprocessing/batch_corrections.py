class Harmony():
    def fit_transform(self, X, y):
        """
        A wrapper of Harmony algorithm implmented in harmonypy package
        see: https://github.com/slowkow/harmonypy

        Args:
            X (ndarray of shape (m, n))): array of data, with m the number of observations in R^n.
            
            y(ndarray of shape (m)): vector of labels for the data. Assumed to be discrete; string or
            other labels are handled cleanly.
        Return:
            (ndarray of shape (m, n))): Modified data matrix.
        """

        import harmonypy as hm
        import pandas as pd
        columns = ['batch_label']
        batch_labels = pd.DataFrame(y.reshape(-1, 1), columns=columns)
        ho = hm.run_harmony(X, batch_labels, columns)
        return ho.result().T



class Limma():
    def fit_transform(self, X, y):
        """
        A pure numpy implementation of the code found at:

        https://github.com/chichaumiau/removeBatcheffect/blob/master/limma.py

        based on our weak understanding of what the patsy package does and
        following along with the example in the link above. Their data comes from

        https://github.com/brentp/combat.py

        Note that we don't have the capability to include an assumed model effect
        in the covariance_matrix as in Chichau's version, but our approach is
        only to remove batch factors, then apply machine learning algo's to the
        result. Making an initial assumption of a linear model in the phenotypes in
        preprocessing stage may not be appropriate depending on the machine learning
        tools used later in the pipeline.

        Args:
            X (ndarray of shape (m, n))): array of data, with m the number of observations in R^n.
            
            y(ndarray of shape (m)): vector of labels for the data. Assumed to be discrete; string or
            other labels are handled cleanly.
        Return:
            (ndarray of shape (m, n))): Modified data matrix.

        There are options in the original limma.removeBatchEffect() code
        and corresponding limma_chichau() function (see in calcom/utils/limma.py)
        which aren't implemented in this version.
        """
        import numpy as np

        m,n = X.shape

        unique_batches = np.unique(y)
        bmap = {b:i for i,b in enumerate(unique_batches)}
        nbatches = len(np.unique(y))

        design_matrix = np.zeros((m,nbatches))
        for i,b in enumerate(y):
            design_matrix[i,bmap[b]] = 1
        #

        # Idea here is that the states of each of the labels is encoded
        # in R^{nbatches-1} where each cardinal direction represents a
        # batch, and the origin is the first batch.
        design_matrix = design_matrix[:,1:]

        # Seems to insert a row of -1's for any sample in the first batch.
        rowsum = design_matrix.sum(axis=1) -1
        design_matrix=(design_matrix.T+rowsum).T

        # Apparently this is the "null" model generated; just a ones vector.
        covariate_matrix = np.ones((m,1))
        design_batch = np.hstack( (covariate_matrix,design_matrix) )

        # coefficients, _, _, _ = np.linalg.lstsq(design_batch,data)
        # Need to silence an annoying warning.
        numpy_version = '.'.join( np.__version__.split('.')[:2] )
        if numpy_version < '1.14':
            coefficients, _, _, _ = np.linalg.lstsq(design_batch,X)
        else:
            coefficients, _, _, _ = np.linalg.lstsq(design_batch,X,rcond=None)
        #

        # Subtract off the component of this least squares linear model
        # whose contribution is due to batch effect.
        return X - np.dot(design_matrix,coefficients[-(nbatches-1):])
    #