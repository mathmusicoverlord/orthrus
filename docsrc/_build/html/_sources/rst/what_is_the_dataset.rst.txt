What is a DataSet object?
===========================

A :py:class:`DataSet <datasci.core.dataset.DataSet>` object is a data container designed to automate statistical, machine learning, and manifold learning tasks including, 
but not limited to:

   * Data pre-processing, e.g., batch correction, normalization, imputation
   * Statistical summarization of data and associated metadata
   * Data visualization, e.g., Principal Component Analysis (PCA), Multi-dimensional Scaling (MDS), 
     Uniform Manifold Projection and Approximation (UMAP)
   * Classification, e.g., Support Vector Machines (SVM), Random Forest (RF), Artificial Neural Networks (ANN)
   * Feature selection

A :py:class:`DataSet <datasci.core.dataset.DataSet>` object is primarily comprised of three data structures\: :py:attr:`data <DataSet.data>` , :py:attr:`metadata <datasci.core.dataset.DataSet.metadata>`, and :py:attr:`vardata <datasci.core.dataset.DataSet.vardata>`

:py:attr:`experiments <datasci.core.dataset.DataSet.experiments>`

A key feature of the :py:class:`DataSet <datasci.core.dataset.DataSet>` object is that ...