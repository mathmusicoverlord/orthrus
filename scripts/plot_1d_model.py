if __name__ == '__main__':
    # imports for arguments
    import argparse
    import os

    # command line arguments
    parser = argparse.ArgumentParser("plot-1d-ML-model")

    parser.add_argument('--exp_params',
                        type=str,
                        default=os.path.join(os.environ['DATASCI_PATH'], 'test_data', 'Iris', 'Experiments',
                                             'setosa_versicolor_classify_species_svm',
                                             'setosa_versicolor_classify_species_svm_params.py'),
                        help='File path of containing the experimental parameters. Default is the Iris experiment.')

    parser.add_argument('--n_orth_comps',
                        type=int,
                        default=1,
                        choices=[1, 2],
                        help='The number of orthogonal components to use in PCA.')

    parser.add_argument('--backend',
                        type=str,
                        default='pyplot',
                        choices=['pyplot', 'plotly'],
                        help='The plotting backend used to generate the plots.')

    args = parser.parse_args()

    # imports
    from sklearn.decomposition import PCA
    from sklearn.metrics import balanced_accuracy_score as bsr
    from datasci.decomposition.general import OrthTransform
    from sklearn.model_selection import LeaveOneGroupOut
    import torch as tc
    import numpy as np
    import sys
    import os
    from copy import deepcopy
    from datasci.core.helper import module_from_path

    # set experiment parameters
    exp_params = module_from_path('exp_params', args.exp_params)
    results_dir = exp_params.RESULTS_DIR
    exp_name = exp_params.EXP_NAME
    class_attr = exp_params.CLASS_ATTR
    train_test_attr = exp_params.TRAIN_TEST_ATTR
    ds = exp_params.DATASET
    sample_ids = exp_params.SAMPLE_IDS
    feature_ids = exp_params.FEATURE_IDS
    classifier_results = exp_params.CLASSIFIER_RESULTS
    classifier_name = exp_params.CLASSIFIER_NAME
    classifier_weights_handle = exp_params.CLASSIFIER_WEIGHTS_HANDLE
    classifier_bias_handle = exp_params.CLASSIFIER_BIAS_HANDLE

    # get number of features
    n_features = len(feature_ids)

    # loop through classification results
    for i, _ in enumerate(classifier_results['classifiers']):
        #grab classifier
        classifier = classifier_results['classifiers'][i]

        # grab scores
        #train_score = classifier_results['scores'].iloc[i]['Train'].item()
        test_score = classifier_results['scores'].loc['Test'].values[i]

        # extract weights vector
        w = eval("classifier." + classifier_weights_handle).reshape(-1, 1)

        # extract bias/shift
        try:
            b = np.array(eval("classifier." + classifier_bias_handle)).item()
        except AttributeError:
            b = 0

        # find a point in the kernel of the model
        x0 = np.matmul(np.linalg.pinv(w.reshape(1, -1)), np.array([[b]]))

        pca = PCA(n_components=args.n_orth_comps, whiten=True, random_state=0)
        class_embed = OrthTransform(subspace=w, shift=x0, transformer=pca)

        if args.backend == 'pyplot':
            if args.n_orth_comps == 1:

                ds.visualize(embedding=class_embed,
                             viz_name=classifier_name,
                             attr=class_attr,
                             cross_attr=train_test_attr,
                             feature_ids=feature_ids,
                             sample_ids=sample_ids,
                             xlabel='Model Axis',
                             ylabel='PC 1 Orth.',
                             palette='bright',
                             alpha=.6,
                             edgecolors='face',
                             mrkr_list=['^', 'o'],
                             s=200,
                             linewidths=0,
                             subtitle="Test Score = " + str(test_score * 100) + "%,    # features = " + str(n_features),
                             save=True,
                             save_name='_'.join([ds.name, exp_name, 'classifier', str(i), 'features', str(n_features), classifier_name,
                                                 class_attr.lower()]))
            else:
                ds.visualize(embedding=class_embed,
                             viz_name=classifier_name,
                             attr=class_attr,
                             cross_attr=train_test_attr,
                             feature_ids=feature_ids,
                             sample_ids=sample_ids,
                             xlabel='Model Axis',
                             ylabel='PC 1 Orth.',
                             zlabel='PC 2 Orth.',
                             palette='bright',
                             alpha=.6,
                             edgecolors='face',
                             mrkr_list=['^', 'o'],
                             s=200,
                             linewidths=0,
                             subtitle="Test Score = " + str(test_score * 100) + "%,    # features = " + str(n_features),
                             save=True,
                             save_name='_'.join([ds.name, exp_name, 'classifier', str(i), 'features', str(n_features), classifier_name,
                                                 class_attr.lower(), '3d']))
        elif args.backend == 'plotly':
            if args.n_orth_comps == 1:
                ds.visualize(embedding=class_embed,
                             viz_name=classifier_name,
                             attr=class_attr,
                             cross_attr=train_test_attr,
                             feature_ids=feature_ids,
                             sample_ids=sample_ids,
                             xlabel='Model Axis',
                             ylabel='PC 1 Orth.',
                             backend='plotly',
                             mrkr_list=['diamond', 'circle'],
                             opacity=.7,
                             figsize=(1500, 1000),
                             subtitle="Test Score = " + str(test_score * 100) + "%,    # features = " + str(n_features),
                             save=True,
                             save_name='_'.join(
                                 [ds.name, exp_name, 'classifier', str(i), 'features', str(n_features), classifier_name,
                                  class_attr.lower()]))
            else:
                ds.visualize(embedding=class_embed,
                             viz_name=classifier_name,
                             attr=class_attr,
                             cross_attr=train_test_attr,
                             feature_ids=feature_ids,
                             sample_ids=sample_ids,
                             xlabel='Model Axis',
                             ylabel='PC 1 Orth.',
                             zlabel='PC 2 Orth.',
                             backend='plotly',
                             mrkr_list=['diamond', 'circle'],
                             opacity=.7,
                             figsize=(1500, 1000),
                             subtitle="Test Score = " + str(test_score * 100) + "%,    # features = " + str(n_features),
                             save=True,
                             save_name='_'.join(
                                 [ds.name, exp_name, 'classifier', str(i), 'features', str(n_features), classifier_name,
                                  class_attr.lower(), '3d']))