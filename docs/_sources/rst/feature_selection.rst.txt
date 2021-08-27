Feature Selection using IFR
===========================
:py:class:`DataSet's <orthrus.core.dataset.DataSet>` through its modular design makes it really easy to perform feature selection using a variety of feature selection methods. This
tutorial provides an in depth look of using :py:meth:`feature_select <orthrus.core.dataset.DataSet.feature_select>` method in combination with :py:class:`IFR <orthrus.sparse.feature_selection.IterativeFeatureRemoval.IFR>`.
    
The first step is to load the dataset and make sure we have the right attribute in the metadata. For this example we are going to work with the GSE730732 dataset and select features for 
controls vs shedders in hour 1 to 8. We resolved the issues with datatypes in GSE73072 dataset in `creating a dataset <create_dataset.html>`_, learnt how to `normalize the data <normalization_and_imputation.html>`_
, and created a new attribute ``response`` in the `adding new attributes to metadata <add_new_attributes_using_queries.html>`_, please check these tutorial first.

    >>> # load dataset
    >>> from orthrus.core.dataset import load_dataset
    >>> ds = load_dataset('path/to/gse73072.ds')
    >>> class_attr = 'Response'
    >>> from orthrus.preprocessing.batch_corrections import Limma
    >>> limma_obj = Limma()
    >>> #Apply Limma on SubjectID attribute
    >>> ds.normalize(limma_obj, norm_name='Limma', supervised_attr='SubjectID')


    >>> controls_query = 'time_id<=0'
    >>> shedders_query = 'time_id> 0 and time_id<=8 and shedding == True'
    >>> qs = {'controls': controls_query, 'shedders': shedders_query}
    >>> new_attribute_name = 'Response'
    >>> ds.generate_attr_from_queries(new_attribute_name, qs, attr_exist_mode='overwrite')
 
In this example we want to extract feature that can distinguish between samples in controls and the shedders in hours 1 to 8. So, first we want to restrict out data to the sample ids we care about, 
which in this case are the samples from these two classes. Further, let's also restrict the samples based on the StudyID, here we will work with the four influenza studies. 
So, we now want to extract features that can distinguish control and shedders from flu studies only.  

The code snippet below does this job. 

    >>> import numpy as np
    >>> studies = np.array(['gse73072_dee2', 'gse73072_dee3', 'gse73072_dee4', 'gse73072_dee5'])
    >>> # restrict samples to two classes and restrict based on StudyIDs
    >>> sample_ids = (ds.metadata[class_attr].isin(['controls', 'shedders'])) & (ds.metadata['StudyID'].isin(studies))

Next, let's define our feature selector, which is Iterative Feature Removal, for our example. First, we need to define a classifier that IFR will use. In this example we are going 
to use GPU based :py:class:`SSVMClassifier <orthrus.sparse.classifiers.svm.SSVMClassifier>` with `LPPrimalDualPy <https://github.com/CSU-PAL-biology/calcom/blob/development/calcom/solvers/LPPrimalDualPy.py>`_ solver

    >>> from orthrus.sparse.classifiers.svm import SSVMClassifier
    >>> from calcom.solvers import LPPrimalDualPy
    >>> model = SSVMClassifier(C = 1, solver =LPPrimalDualPy, use_cuda = True)
    >>> weights_handle="weights_"

Second, let's create the :py:class:`IFR <orthrus.sparse.feature_selection.IterativeFeatureRemoval.IFR>` object. Please check the documentation to understand about the arguments to IFR.

    >>> from orthrus.sparse.feature_selection.IterativeFeatureRemoval import IFR
    >>> feature_selector = IFR(model,
    ...                        weights_handle=weights_handle,
    ...                        verbosity = 2,
    ...                        nfolds = 5,
    ...                        repetition = 1,
    ...                        cutoff = .80,
    ...                        jumpratio = 5,
    ...                        max_iters = 20,
    ...                        max_features_per_iter_ratio = 0.8,
    ...                        num_gpus_per_worker=0.1
    ...                        )


Third, we now define the parameters for the :py:meth:`feature_select <orthrus.core.dataset.DataSet.feature_select>` method. Please check the method documentation to know the parameters in detail.
Here, we are interested in the following attributes: ``selector``, ``attr``, ``selector_name``, ``sample_ids``, ``fit_handle``, ``f_results_handle``. So far we have defined all but ``fit_handle`` and ``f_results_handle``.
The purpose of these arguments is to provide alternate handle to the feature selectors ``fit`` or ``run`` method and the handle on how to access the results. Although the default values of 
``fit_handle``, which is 'fit', matches the fit handle of IFR, ``f_results_handle`` does not. We now show how to provide a different results handle.

For IFR class, the results after the feature selection can be accessed by accessing the results attribute as shown below.

    >>> feature_selector.results

So, we set the ``f_results_handle`` to 'results'.

    >>> feature_selection_results_handle = 'results'

At this stage we have all the varibles we need for feature selection and we are ready to run feature selection.

    >>> feature_selection_results = ds.feature_select(feature_selector,
    ...                             attr=class_attr,
    ...                             selector_name='ifr',
    ...                             sample_ids=sample_ids,
    ...                             f_results_handle=feature_selection_results_handle
    ...                             )
    
The return of :py:meth:`feature_select <orthrus.core.dataset.DataSet.feature_select>` method is a dictionary that contains two elements:

 1. ``selector`` : This is ``feature_selector`` object and can be used now to access any information about the feature selection.
 2. ``f_results`` : This a Pandas.DataFrame with feature ids (columns of ds.data) as index and columns are ``feature_selector`` specific. For 
    instance, ``f_results`` for ``IFR``  contains three columns ``frequency``, ``weights`` and ``selection_iteration`` for each feature id.

Finally, we can save these to disk using :py:meth:`save_object <orthrus.core.helper.save_object>` method.

    >>> from orthrus.core.helper import save_object
    >>> save_object(feature_selection_results, 'path/to/dst/dir/control_vs_shedders/feature_selection_results.pickle')

In the `next tutorial <feature_set_size_reducton.html>`_ we will look at how to reduce the size of these features in ``f_results`` to find an optimal number of features for a classification problem.