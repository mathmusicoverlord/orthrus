Feature Set Size Reduction
===========================

In the `previous tutorial <feature_selection.html>`_, we learnt how to select features using the :py:meth:`feature_select <datasci.core.dataset.DataSet.feature_select>` method 
in combination with :py:class:`IFR <datasci.sparse.feature_selection.IterativeFeatureRemoval.IFR>`. However, these feature sets can often be quite large and are often not ready
to be used for classification just yet. For example, out of the 22,277 features in the GSE73072 dataset, the ``IFR`` algorithm may extract around 5000-8000 features, 
i.e. features with ``frequency > 0``. So, in this step we are aiming to reduce the size of this feature set using :py:meth:`reduce_feature_set_size <datasci.sparse.feature_selection.helper.reduce_feature_set_size>` method


LOAD THE DATASET OBJECT AND DEFINE THE QUERIES AGAIN

Let's first load the object we saved to the disk in the last tutorial using :py:meth:`load_object <datasci.core.helper.load_object>` method.

    >>> from datasci.core.helper import load_object
    >>> feature_selection_results = load_object('path/to/dst/dir/control_vs_shedders/feature_selection_results.pickle')


The feature set size reduction is essentially a grid search on the number of features. In this we first rank our features by a method, as we will see below. Then, we 
define a grid of values for the number of top ranked features to use in the upcoming step. Next, for each sampled value `n` from the grid, we use the top `n` feature to train and evaluate a new model.
Please check the method docomentation for :py:meth:`reduce_feature_set_size <datasci.sparse.feature_selection.helper.reduce_feature_set_size>` method to see how to use ``partitioners`` and/or ``train/validation`` partition splits.
These settings control on which partition the score is evaluated on.


To use :py:meth:`reduce_feature_set_size <datasci.sparse.feature_selection.helper.reduce_feature_set_size>` method, we need to define a ``model``, ``scorer``, and ``partitioner`` and the grid itself.

    >>> from sklearn.metrics import balanced_accuracy_score
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.model_selection import KFold
    >>> model = LinearSVC(dual=False)
    >>> scorer = balanced_accuracy_score
    >>> partitioner = KFold(n_splits=5, shuffle=True, random_state=0)
    >>> #define the grid of values for top features
    >>> start = 1
    >>> end = 50
    >>> step = 1

Next, we are going need a ranking scheme to rank for our features. :py:mod:`feature_helper <datasci.sparse.feature_selection.helper>` module
provides a few ranking methods. In this tutorial, we are going to use :py:meth:`rank_features_by_attribute <datasci.sparse.feature_selection.helper.rank_features_by_attribute>` to 
rank our features. This method provides the functionality to rank features by a numeric column in ``feature_selection_results['f_results']`` in ascending or descending order. 
As discussed briefly in the `previous tutorial <feature_selection.html>`_, ``IFR's`` result contains three columns and one of them is ``frequency``, which we are going to use
to rank our features. A feature with higher frequency is ranked higher. 

We need to define the ranking method and the arguments to the ranking method:



    >>> from  datasci.sparse.feature_selection.helper import rank_features_by_attribute
    >>> ranking_method = rank_features_by_attribute
    >>> # args to rank feature by frequency in descending order and restricting to features that have frequency greater than 5
    >>> restricted_features_ids = (feature_selection_results['f_results']['frequency'] > 5)
    >>> ranking_method_args = {'attr': 'frequency', 'order': 'desc', 'feature_ids': restricted_features_ids}



TALK ABOUT RAY PROCESSES


    reduced_feature_results = reduce_feature_set_size(ds, 
                            features, 
                            sample_ids,
                            attr,
                            model, 
                            scorer, 
                            ranking_method,
                            ranking_method_args,
                            partitioner=partitioner,
                            start=start, 
                            end=end, 
                            step=step,
                            num_cpus_per_worker=num_cpus_per_worker,
                            num_gpus_per_worker=num_gpus_per_worker)


TALK ABOUT RESULTS 


SAVE RESULTS TO DISK