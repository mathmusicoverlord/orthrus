Data Normalization and Imputation
=================================

The :py:class:`DataSet <orthrus.core.dataset.DataSet>` class makes it convinient to normalize and impute data by providing methods such as
:py:meth:`normalize <orthrus.core.dataset.DataSet.normalize>` and
:py:meth:`impute <orthrus.core.dataset.DataSet.impute>`. These methods are compatible with various normalization and imputation methods available
in `sklearn <https://scikit-learn.org/stable/index.html>`_ package. Please check the method documentation for more details and how to write custom 
normalization and imputation methods

Normalization
-------------
The dataset package provides various normalization (:py:class:`LogNormalizer <orthrus.preprocessing.normalizations.LogNormalizer>` and
:py:class:`MedianFoldChangeNormalizer <orthrus.preprocessing.normalizations.MedianFoldChangeNormalizer>`) and batch correction normlizations
such as :py:class:`Limma <orthrus.preprocessing.batch_corrections.Limma>` and :py:class:`Harmony <orthrus.preprocessing.batch_corrections.Harmony>`.

:py:class:`LogNormalizer <orthrus.preprocessing.normalizations.LogNormalizer>`, which performs a element wise log operations is an unsupervised
normalization method. Let's take a look on how to use it.
::
    >>> from orthrus.core.dataset import load_dataset
    >>> from orthrus.preprocessing.normalizations import LogNormalizer

    >>> ds = load_dataset('/path/to/ds')
    >>> normalization_method = 'log'
    >>> log_normalizer = LogNormalizer()

    >>> ds.normalize(log_normalizer, norm_name=normalization_method)
    >>> #updated data 
    >>> print('normalized data', ds.data)

:py:meth:`normalize <orthrus.core.dataset.DataSet.normalize>` method updates the :py:attr:`data <orthrus.core.dataset.DataSet.data>` after performing the normalization operation. Next,
let's take a look a look at how to use a batch correction normalization method, say Limma. Since these methods are supervised, in addition to the data matrix :py:meth:`normalize <orthrus.core.dataset.DataSet.normalize>`
method takes `supervised_attr` parameter (the name of attribute in :py:attr:`metadata <orthrus.core.dataset.DataSet.metadata>`) as additional input; the label information for this attribute is used for normalization.
::
    >>> from orthrus.preprocessing.batch_corrections import Limma
    >>> limma = Limma()
    >>> ds.normalize(limma, norm_name='Limma', supervised_attr='BatchID') #ds.metadata must contain BatchID attribute

Imputation
----------
Similar to :py:meth:`normalize <orthrus.core.dataset.DataSet.normalize>` method the :py:meth:`impute <orthrus.core.dataset.DataSet.impute>` method may be used to impute missing values. As described
at the beginning of the document, this method can work with a variety of imputation methods from packages like `sklearn <https://scikit-learn.org/stable/index.html>`_ . Please see the method documentation
on the restrictions and how to write custom imputation methods.

Let's look at an example. Suppose that our data contains missing values, which are represented as zero in this case, and we want to impute them using `KNNImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html>`_ from 
`sklearn` package.
::
    >>> impute_name='knn'
    >>> imputer = KNNImputer(missing_values=0)
    >>> ds.impute(imputer, impute_name=impute_name)

Similar to :py:meth:`normalize <orthrus.core.dataset.DataSet.normalize>` method, the :py:meth:`impute <orthrus.core.dataset.DataSet.impute>` method updates the :py:attr:`data <orthrus.core.dataset.DataSet.data>` after performing the normalization operation.