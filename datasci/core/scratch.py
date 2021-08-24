if __name__ == "__main__":

    from datasci.core.pipeline import *
    from sklearn.model_selection import KFold
    from sklearn.model_selection import ShuffleSplit
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from datasci.core.helper import save_object, load_object
    from datasci.sparse.feature_selection.kffs import KFFS
    from datasci.sparse.classifiers.svm import SSVMClassifier as SSVM
    from datasci.manifold.mds import MDS
    from calcom.solvers import LPPrimalDualPy
    import ray
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import FunctionTransformer

    """
    This file contains the experimental constants for the experiment classify_setosa_versicolor_svm.
    All experimental parameters to be exported are denoted by UPPERCASE names as a convention.
    """

    # imports
    import datetime
    import os
    from datasci.core.dataset import load_dataset
    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import LeaveOneOut
    from sklearn.model_selection import KFold

    # load dataset
    ds = load_dataset('/hdd/DataSci/test_data/GSE73072/Data/GSE73072.ds')
    ds = ds.slice_dataset(feature_ids=ds.vardata.index[:100])

    # define train/test partition object
    tr_tst_80_20 = Partition(process=ShuffleSplit(n_splits=1,
                                                  random_state=16575,
                                                  ),
                             process_name='80-20',
                             )

    # define inner train/validation partition object
    kfold = Partition(process=KFold(n_splits=15,
                                    shuffle=True,
                                    random_state=124,
                                    ),
                      process_name='15-fold-CV',
                      #parallel=True,
                      verbosity=1,
                      )

    # define leave one out
    loo = Partition(process=LeaveOneOut(),
                    process_name='LOO',
                    #parallel=True,
                    verbosity=2)

    # define log transform
    log = Transform(process=FunctionTransformer(np.log),
                    process_name='log',
                    retain_f_ids=True,
                    #parallel=True,
                    verbosity=1)

    # define polynomial transform
    quad = Transform(process=FunctionTransformer(lambda x: np.power(x, 2) - x + 1),
                     process_name='quad',
                     retain_f_ids=True,
                     #parallel=True,
                     verbosity=1)

    # define MDS transform
    mds = Transform(process=MDS(n_components=500),
                    process_name='mds',
                    #parallel=True,
                    )

    # define PCA transform
    pca = Transform(process=PCA(n_components=3, whiten=True),
                    process_name='pca',
                    #parallel=True,
                    verbosity=1)

    # define LinearSVC classify process
    svm = Classify(process=LinearSVC(),
                   process_name='svm',
                   class_attr='Shedding',
                   #parallel=True,
                   verbosity=1,
                   )

    # define LinearSVC classify process
    rf = Classify(process=RandomForestClassifier(),
                  process_name='RF',
                  class_attr='Shedding',
                  #parallel=True,
                  verbosity=1,
                  f_weights_handle='feature_importances_',
                  )

    # define kfold feature selection process
    kffs = FeatureSelect(process=KFFS(classifier=SSVM(solver=LPPrimalDualPy,
                                                      use_cuda=True),
                                      f_weights_handle='weights_',
                                      f_rnk_func=np.abs,
                                      random_state=235,
                                      ),
                          process_name='kffs',
                          #parallel=True,
                          supervised_attr='Shedding',
                          transform_args=dict(n_top_features=10),
                          f_ranks_handle='ranks_',
                          verbosity=1)

    # define bsr score
    bsr = Score(process=balanced_accuracy_score,
                process_name='bsr',
                pred_attr='Shedding',
                #parallel=True,
                verbosity=2,
                )

    # create pipeline to run
    pipeline = Pipeline(processes=(tr_tst_80_20, pca, kfold, quad, rf, bsr),
                        pipeline_name='test',
                        verbosity=1)

    # initiate ray for parallel processsion
    #ray.init(_temp_dir="/hdd/tmp/ray/")

    # run pipeline
    pipeline.run(ds)

    # get results
    results = pipeline.results_

    ray.shutdown()

