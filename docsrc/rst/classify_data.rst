Classification
==============

In this tutorial we will learn how to setup and perform classification experiments using the
:py:meth:`classify() <datasci.core.dataset.DataSet.classify>` method of the 
:py:class:`DataSet <datasci.core.dataset.DataSet>` class. We start by loading the Iris dataset
provided in the DataSci package::

    >>> # imports
    >>> import os
    >>> from datasci.core.dataset import load_dataset

    >>> # load the data
    >>> file_path = os.path.join(os.environ["DATASCI_PATH"],
    ...                          "test_data/Iris/Data/iris.ds")
    >>> ds = load_dataset(file_path)

Train/Test Split with Linear SVM
--------------------------------

Our first experiment will be simple. Train a 
`linear support vector machine classifier <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html>`_
(SVM) on 80% of the data to distinguish between the *setosa* and *versicolor* iris species, test the trained SVM model
on the left over 20%, and record the 
`balanced success rate <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html>`_ (bsr)::

    >>> # imports
    >>> from sklearn.metrics import balanced_accuracy_score as bsr
    >>> from sklearn.model_selection import ShuffleSplit
    >>> from sklearn.svm import LinearSVC

    >>> # define the sample ids
    >>> setosa_versicolor = ds.metadata['species'].isin(['setosa', 'versicolor'])

    >>> # define the partitioner
    >>> train_test_part = ShuffleSplit(random_state=0,  # for reproducibility
    ...                                train_size=.8,   # test_size=.2 implicitly,
    ...                                n_splits=1,
    ...                                )

    >>> # define the classifier
    >>> svm = LinearSVC()

    >>> # run the classification experiment
    >>> classification_results = ds.classify(classifier=svm,
    ...                                      classifier_name='SVM',
    ...                                      attr='species',  # what you are classifying
    ...                                      sample_ids=setosa_versicolor,
    ...                                      partitioner=train_test_part,
    ...                                      partitioner_name='80Train_20Test',
    ...                                      scorer=bsr,
    ...                                      scorer_name='BSR',
    ...                                      )
    
    SVM, Split 1 of 1, Scores: 
    ===========================
    Training BSR: 100.00%
    ---------------------------
    Test BSR: 100.00%
    
    SVM, Summary, Scores: 
    ======================
    Training BSR: 100.00% +/- nan%
    Max. Training BSR: 100.00%
    Min. Training BSR: 100.00%
    ----------------------
    Test BSR: 100.00% +/- nan%
    Max. Test BSR: 100.00%
    Min. Test BSR: 100.00%

The classification results are represented as a dictionary containing the classifiers,
classification scores, partitioning and classification labels, classifier feature weights,
and the classifier sample weights respectively. To obtain the feature weights or sample weights
of the classification model we have to let the classify method know where the feature or sample weights reside
in the classifier object by specifying either the ``f_weights_handle`` and/or ``s_weights_handle`` parameter. 
For the `LinearSVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html>`_
the feature weights reside in the ``coef_`` attribute. Here is an example where we extract the feature weights by adjusting
the above code slightly:: 

    >>> # classify (/w extracting the model weights)
    >>> classification_results = ds.classify(classifier=svm,
    ...                                      classifier_name='SVM',
    ...                                      attr='species',  # what you are classifying
    ...                                      sample_ids=setosa_versicolor,
    ...                                      partitioner=train_test_part,
    ...                                      partitioner_name='80Train_20Test',
    ...                                      scorer=bsr,
    ...                                      scorer_name='BSR',
    ...                                      f_weights_handle='coef_',
    ...                                      )

    >>> # print the feature weights
    >>> print(classification_results['f_weights'])

                  species_80Train_20Test_SVM_f_weights_0  species_80Train_20Test_SVM_f_rank_0
    sepal_length                               -0.213456                                    2
    sepal_width                                -0.403911                                    3
    petal_length                                0.831287                                    0
    petal_width                                 0.413565                                    1

We can see the largest weight feature in the SVM model is the ``petal_length`` of the flower,
this is reflected in the rank of the feature.

GSE7302: k-Fold Cross-Validation with SSVM on GPU
--------------------------------------------------

In this example we will run a k-fold cross-validation experiment with the GSE73072 dataset
using :py:class:`Sparse Support Vector Machines <datasci.sparse.classifiers.svm.SSVMClassifier>` (SSVM)
on a GPU to distguish between shedding and non-shedding individuals infected with HRV in the 
first 24 hours of exposure. First we load the dataset::

    >>> # imports
    >>> import os
    >>> from datasci.core.dataset import load_dataset

    >>> # load the data
    >>> file_path = os.path.join(os.environ["DATASCI_PATH"],
    ...                          "test_data/GSE73072/Data/GSE73072.ds")
    >>> ds = load_dataset(file_path)

Then we set the sample ids, partitioner, and the classifier::

    >>> # imports
    >>> from datasci.sparse.classifiers.svm import SSVMClassifier as SSVM
    >>> from sklearn.model_selection import StratifiedKFold
    >>> from calcom.solvers import LPPrimalDualPy

    >>> # define the sample ids
    >>> hrv_first_24 = ds.metadata.query("virus=='HRV' &  0<=time_point_hr<=24").index

    >>> # define the partitioner
    >>> kfold = StratifiedKFold(random_state=0,
    ...                         n_splits=5)

    >>> # define the classifier
    >>> ssvm = SSVM(C=1,
    ...             use_cuda=True,  # Using the GPU
    ...             solver=LPPrimalDualPy)


We now run the experiment::

    >>> # run the classification experiment
    >>> classification_results = ds.classify(classifier=ssvm,
    ...                                      classifier_name='SSVM',
    ...                                      attr='Shedding',
    ...                                      sample_ids=hrv_first_24,
    ...                                      partitioner=kfold,
    ...                                      partitioner_name='5-Fold',
    ...                                      scorer=bsr,
    ...                                      scorer_name='BSR',
    ...                                      f_weights_handle='weights_',
    ...                                      groups='virus',  # passed to the StratifiedKFold.split() for balanced splitting
    ...                                      )

    SSVM, Split 1 of 5, Scores: 
    ============================
    Training BSR: 100.00%
    ----------------------------
    Test BSR: 57.25%
    
    SSVM, Split 2 of 5, Scores: 
    ============================
    Training BSR: 100.00%
    ----------------------------
    Test BSR: 51.25%
    
    SSVM, Split 3 of 5, Scores: 
    ============================
    Training BSR: 100.00%
    ----------------------------
    Test BSR: 60.75%
    
    SSVM, Split 4 of 5, Scores: 
    ============================
    Training BSR: 100.00%
    ----------------------------
    Test BSR: 81.75%
    
    SSVM, Split 5 of 5, Scores: 
    ============================
    Training BSR: 100.00%
    ----------------------------
    Test BSR: 57.90%
    
    SSVM, Summary, Scores: 
    =======================
    Training BSR: 100.00% +/- 0.00%
    Max. Training BSR: 100.00%
    Min. Training BSR: 100.00%
    -----------------------
    Test BSR: 61.78% +/- 11.69%
    Max. Test BSR: 81.75%
    Min. Test BSR: 51.25%