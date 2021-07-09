What is the DataSet object?
===========================

The :py:class:`DataSet <datasci.core.dataset.DataSet>` object is a data container designed to automate statistical, machine learning, 
and manifold learning tasks including, but not limited to:

* Data pre-processing, e.g., batch correction, normalization, imputation
* Statistical summarization of data and associated metadata
* Data visualization, e.g., Principal Component Analysis (PCA), Multi-dimensional Scaling (MDS), 
  Uniform Manifold Projection and Approximation (UMAP)
* Classification, e.g., Support Vector Machines (SVM), Random Forest (RF), Artificial Neural Networks (ANN)
* Feature selection

The :py:class:`DataSet <datasci.core.dataset.DataSet>` object is primarily composed of three
data structures\: :py:attr:`data <datasci.core.dataset.DataSet.data>`, :py:attr:`metadata <datasci.core.dataset.DataSet.metadata>`,
and :py:attr:`vardata <datasci.core.dataset.DataSet.vardata>`, each of which is a 
`Pandas.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ object containing an indexed rectangular array of values. 
These dataframes are described as follows:

* :py:attr:`data <datasci.core.dataset.DataSet.data>` \: The rows of the ``DataFrame`` respresent samples, e.g., participants in a clinical study.
  The columns represent the features, or observations, of each sample e.g., RNA seq expression values, metabolite peak areas for given *m/z* s
  and retention times, or shedding scores across multiple virus and bacteria. The rows of the data are labeled via the ``index`` of the ``DataFrame``
  and the columns are labeled via the ``columns`` of the ``DataFrame``.

A key feature of the :py:class:`DataSet <datasci.core.dataset.DataSet>` object is that ...blah blah blah