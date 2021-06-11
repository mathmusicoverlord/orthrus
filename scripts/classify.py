"""
Generic script for classifying a dataset.
"""

if __name__ == '__main__':
    # imports for arguments
    import argparse
    import os

    # command line arguments
    parser = argparse.ArgumentParser("generic-classification")

    parser.add_argument('--exp_params',
                        type=str,
                        default=os.path.join(os.environ['DATASCI_PATH'], 'test_data', 'Iris', 'Experiments',
                                             'setosa_versicolor_classify_species_svm',
                                             'setosa_versicolor_classify_species_svm_params.py'),
                        help='File path of containing the experimental parameters. Default is the Iris experiment.')

    parser.add_argument('--score',
                        type=str,
                        default='bsr',
                        choices=['bsr', 'accuracy', 'confusion'],
                        help='Score reported on the prediction results.')

    args = parser.parse_args()

    # imports
    from datasci.core.helper import save_object
    from datasci.core.helper import module_from_path
    from datasci.core.helper import default_val

    # set experiment parameters
    exp_params = module_from_path('exp_params', args.exp_params)
    script_args = exp_params.CLASSIFIER_ARGS

    ## required script params
    results_dir = script_args.get('RESULTS_DIR', exp_params.RESULTS_DIR)
    exp_name = script_args.get('EXP_NAME', exp_params.EXP_NAME)
    class_attr = script_args.get('CLASS_ATTR', exp_params.CLASS_ATTR)
    ds = script_args.get('DATASET', exp_params.DATASET)
    classifier = script_args.get('CLASSIFIER', exp_params.CLASSIFIER)

    ## optional script params
    sample_ids = script_args.get('SAMPLE_IDS',  default_val(exp_params, 'SAMPLE_IDS')),
    feature_ids = script_args.get('FEATURE_IDS', default_val(exp_params, 'FEATURE_IDS'))
    classifier_name = script_args.get('CLASSIFIER_NAME', default_val(exp_params, 'CLASSIFIER_NAME'))
    classifier_fweights_handle = script_args.get('CLASSIFIER_FWEIGHTS_HANDLE',
                                                 default_val(exp_params, 'CLASSIFIER_FWEIGHTS_HANDLE'))
    classifier_sweights_handle = script_args.get('CLASSIFIER_SWEIGHTS_HANDLE',
                                                 default_val(exp_params, 'CLASSIFIER_SWEIGHTS_HANDLE'))
    partitioner = script_args.get('PARTITIONER', default_val(exp_params, 'PARTITIONER'))
    partitioner_name = script_args.get('PARTITIONER_NAME', default_val(exp_params, 'PARTITIONER_NAME'))
    results_file_name = script_args.get('RESULTS_FILE_NAME',
                                        default_val(exp_params, 'CLASSIFICATION_RESULTS_FILE_NAME'))

    # set scorer
    if args.score == 'bsr':
        from sklearn.metrics import balanced_accuracy_score
        scorer = balanced_accuracy_score
    elif args.score == 'confusion':
        from sklearn.metrics import confusion_matrix
        scorer = confusion_matrix
    elif args.score == 'accuracy':
        from sklearn.metrics import accuracy_score
        scorer = accuracy_score

    # classify data
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

    # save classification results
    if results_file_name is None:
        save_object(classification_results,
                    os.path.join(results_dir,
                                 '_'.join([ds.name, exp_name, args.score, 'classification_results.pickle'])))
    else:
        save_object(classification_results, os.path.join(results_dir, results_file_name))
