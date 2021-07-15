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

* :py:attr:`metadata <datasci.core.dataset.DataSet.metadata>` \: The rows of the ``DataFrame`` represent samples. The columns represent
  decriptive data for each sample, e.g., class label, age, time point, species.

* :py:attr:`vardata <datasci.core.dataset.DataSet.vardata>` \: The rows of the ``DataFrame`` represent features, or observations. The columns represent
  descriptive data for each feature, e.g., location on a chromosome of a gene (gene locus), retention time of a measured metabolite, description of a bacteria. 

The main goal of the :py:class:`DataSet <datasci.core.dataset.DataSet>` object is promote modularity and compatibility with other
data science and machine learning packages, e.g., `sklearn <https://scikit-learn.org/stable/>`_. For example, if a user wishes to visualize their dataset,
rather than hard code or wrap a specific visualization algorithm into the :py:class:`DataSet <datasci.core.dataset.DataSet>` class to make it available to them,
they would pass an embedding object, such as `PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_, to the
:py:meth:`visualize() <datasci.core.dataset.DataSet.visualize>` method which will apply the specific visualization method for the user. The main utility of
the visualization method is to take care of the boiler plate code associated with applying the embedding class, such as generating labels and grabbing the 
data matrix, and plot generation. See the example below::

  # imports
  import os
  from datasci.core.dataset import load_dataset
  from sklearn.decomposition import PCA

  # load dataset
  ds = load_dataset(os.path.join(os.environ['DATASCI_PATH'],
                                'test_data/Iris/Data/iris.ds'))

  # define embedding
  pca = PCA(n_components=2, whiten=True)

  # visualize species of iris with pca
  ds.visualize(embedding=pca,
               attr='species',
               save=True,
               save_name='iris_species_pca')

Visit the `Visualizing Data <visualize_data.html>`_ module 