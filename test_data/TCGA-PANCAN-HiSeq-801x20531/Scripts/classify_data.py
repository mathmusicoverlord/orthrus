"""This script is to provide an example pipeline for new users."""


if __name__ == "__main__":
    # imports
    import os
    from orthrus.core.dataset import load_dataset

    # load dataset
    ds = load_dataset(os.path.join(os.environ['ORTHRUS_PATH'],
                                   "test_data\\TCGA-PANCAN-HiSeq-801x20531\\Data\\TCGA-PANCAN-HiSeq-801x20531-log2.ds"))

    # restrict samples
    sample_ids = ds.metadata.query("tumor_class in ['COAD', 'LUAD']").index
    ds = ds.slice_dataset(sample_ids=sample_ids)

    # define kfold partition
    from sklearn.model_selection import KFold
    from orthrus.core.pipeline import Partition
    kfold = Partition(process=KFold(n_splits=5,
                                    shuffle=True,
                                    random_state=3458,
                                    ),
                      process_name='5-fold-CV',
                      verbosity=1,
                      )

    # define standardization
    from sklearn.preprocessing import StandardScaler
    from orthrus.core.pipeline import Transform
    std = Transform(process=StandardScaler(),
                    process_name='std',
                    retain_f_ids=True,
                    )

    # define feature selector
    from orthrus.solvers.linear import LPPrimalDualPy
    from orthrus.sparse.classifiers.svm import SSVMSelect
    from orthrus.core.pipeline import FeatureSelect
    ssvm = FeatureSelect(process=SSVMSelect(solver=LPPrimalDualPy),
                         process_name='ssvm',
                         supervised_attr='tumor_class',
                         f_ranks_handle='f_ranks',
                         )

    # define classifier
    from sklearn.svm import LinearSVC
    from orthrus.core.pipeline import Classify
    svc = Classify(process=LinearSVC(class_weight='balanced'),
                   process_name='svc',
                   class_attr='tumor_class',
                   f_weights_handle='coef_'
                   )

    # define report
    from orthrus.core.pipeline import Report
    report = Report(pred_attr='tumor_class')

    # define pipeline
    from orthrus.core.pipeline import Pipeline
    pipeline = Pipeline(processes=(kfold, std, ssvm, svc, report),
                        pipeline_name='kfold_std_ssvm_svc',
                        checkpoint_path=os.path.join(os.environ['ORTHRUS_PATH'],
                                                     "test_data/TCGA-PANCAN-HiSeq-801x20531/" \
                                                     "Pipelines/kfold_std_ssvm_svc.pickle"),
                        parallel=True)

    # start the ray server
    import ray
    ray.init()

    # run the pipeline up until the reporting
    # and save the pipeline after each completed process
    pipeline.run(ds, stop_before='report', checkpoint=True)

    # complete the pipeline
    pipeline.run(ds, checkpoint=True)

    # load the pipeline (not needed, just for example)
    from orthrus.core.helper import load_object
    pipeline = load_object(os.path.join(os.environ['ORTHRUS_PATH'],
                                        "test_data/TCGA-PANCAN-HiSeq-801x20531/" \
                                        "Pipelines/kfold_std_ssvm_svc.pickle"))

    # report test statistics
    import numpy as np
    from matplotlib import pyplot as plt
    test_scores = report.report()['train_test'].filter(regex="^((?!Support).)*$").filter(regex="Test")
    test_scores.columns = test_scores.columns.str.strip("Test_")
    print(test_scores)

    # show top 10 features for batch 0 against the other batches
    feature_ranks = pipeline.processes[2].collapse_results(which='f_ranks')['f_ranks']
    batch_0_ranks = feature_ranks["batch_0_Ranks_f_ranks"].argsort().values
    feature_ranks.filter(regex='Ranks').iloc[batch_0_ranks[:10]]

    # show attributes for batch 0 features
    batch_0_attrs = feature_ranks.filter(regex='batch_0').iloc[batch_0_ranks]

