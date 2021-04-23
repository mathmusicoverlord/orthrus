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

    args = parser.parse_args()

    # imports
    from sklearn.metrics import balanced_accuracy_score as bsr
    from datasci.core.helper import save_object
    from datasci.core.helper import module_from_path

    # set experiment parameters
    exp_params = module_from_path('exp_params', args.exp_params)
    results_dir = exp_params.RESULTS_DIR
    exp_name = exp_params.EXP_NAME
    class_attr = exp_params.CLASS_ATTR
    ds = exp_params.DATASET
    sample_ids = exp_params.SAMPLE_IDS
    classifier = exp_params.CLASSIFIER
    classifier_name = exp_params.CLASSIFIER_NAME
    classifier_weights_handle = exp_params.CLASSIFIER_WEIGHTS_HANDLE
    partitioner = exp_params.PARTITIONER
    partitioner_name = exp_params.PARTITIONER_NAME

    # classify data
    classification_results = ds.classify(classifier=classifier,
                                         classifier_name=classifier_name,
                                         attr=class_attr,
                                         sample_ids=sample_ids,
                                         partitioner=partitioner,
                                         partitioner_name=partitioner_name,
                                         scorer=bsr,
                                         scorer_name='bsr',
                                         f_weights_handle=classifier_weights_handle)

    # save classification results
    save_object(classification_results, os.path.join(results_dir, '_'.join([ds.name, exp_name, 'classification_results.pickle'])))
