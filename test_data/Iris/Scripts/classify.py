"""
This script classifies the Iris data set.
"""

if __name__ == '__main__':

    # imports
    import sys
    from sklearn.metrics import balanced_accuracy_score as bsr
    from orthrus.core.helper import save_object

    # set experiment parameters
    sys.path.append()
    from Experiments.setosa_versicolor_classify_species_svm import setosa_versicolor_classify_species_svm_params as exp_params
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
    save_object(classification_results, results_dir + '_'.join([ds.name, exp_name, 'classification_results.pickle']))
