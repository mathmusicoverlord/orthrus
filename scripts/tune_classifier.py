"""
Generic script for tuning a classifier with ray.
"""

if __name__ == '__main__':
    # imports for arguments
    import argparse
    import os

    # command line arguments
    parser = argparse.ArgumentParser("generic-classifier-tuner")

    parser.add_argument('--exp_params',
                        type=str,
                        default=os.path.join(os.environ['DATASCI_PATH'], 'test_data', 'Iris', 'Experiments',
                                             'setosa_versicolor_classify_species_svm',
                                             'setosa_versicolor_classify_species_svm_params.py'),
                        help='File path of containing the experimental parameters. Default is the Iris experiment.')

    parser.add_argument('--score',
                        type=str,
                        default='bsr',
                        choices=['bsr', 'accuracy'],
                        help='Score reported by the classifier.')

    args = parser.parse_args()

    # imports
    from datasci.core.helper import save_object
    from datasci.core.helper import module_from_path
    from ray import tune
    import ray

    # set experiment parameters
    exp_params = module_from_path('exp_params', args.exp_params)
    results_dir = exp_params.RESULTS_DIR
    exp_name = exp_params.EXP_NAME
    class_attr = exp_params.CLASS_ATTR
    ds = exp_params.DATASET
    sample_ids = exp_params.SAMPLE_IDS
    feature_ids = exp_params.FEATURE_IDS
    classifier = exp_params.CLASSIFIER
    classifier_name = exp_params.CLASSIFIER_NAME
    classifier_fweights_handle = exp_params.CLASSIFIER_FWEIGHTS_HANDLE
    classifier_sweights_handle = exp_params.CLASSIFIER_SWEIGHTS_HANDLE
    partitioner = exp_params.PARTITIONER
    partitioner_name = exp_params.PARTITIONER_NAME
    classifier_tuning_params = exp_params.CLASSIFIER_TUNING_PARAMS

    # set scorer
    if args.score == 'bsr':
        from sklearn.metrics import balanced_accuracy_score
        scorer = balanced_accuracy_score
    elif args.score == 'accuracy':
        from sklearn.metrics import accuracy_score
        scorer = accuracy_score

    ray.init(local_mode=True)

    def objective_function(**kwargs):

        # update classifier args
        for key in kwargs:
            classifier.__setattr__(key, kwargs[key])

        classification_results = ds.classify(classifier=classifier,
                                             classifier_name=classifier_name,
                                             attr=class_attr,
                                             sample_ids=sample_ids,
                                             feature_ids=feature_ids,
                                             partitioner=partitioner,
                                             partitioner_name=partitioner_name,
                                             scorer=scorer,
                                             scorer_name=args.score,
                                             f_weights_handle=classifier_fweights_handle,
                                             s_weights_handle=classifier_sweights_handle)

        # grab training score
        return classification_results['scores'].loc['Train'].to_list()

    def trainable(config):
        scores = objective_function(**config)
        [tune.report(score=score) for score in scores]

    tune.run(trainable, config=classifier_tuning_params)

    # save classification results
    #save_object(classification_results, os.path.join(results_dir, '_'.join([ds.name, exp_name, args.score, 'classification_results.pickle'])))
