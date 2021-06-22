"""
Generic script for determining the number of features selected by a sparse classifier.
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

    args = parser.parse_args()

    # imports
    import shutil
    import numpy as np
    from datasci.core.helper import module_from_path
    from datasci.core.helper import default_val
    from sklearn.metrics import balanced_accuracy_score as bsr
    from matplotlib import pyplot as plt

    # set experiment parameters
    exp_params = module_from_path('exp_params', args.exp_params)
    script_args = exp_params.PLOT_FEATURES_OVER_PARAMETER_ARGS

    ## required script parameters
    fig_dir = script_args.get('FIG_DIR', exp_params.FIG_DIR)
    exp_name = script_args.get('EXP_NAME', exp_params.EXP_NAME)
    class_attr = script_args.get('CLASS_ATTR', exp_params.CLASS_ATTR)
    ds = script_args.get('DATASET', exp_params.DATASET)
    classifier = script_args.get('CLASSIFIER', exp_params.CLASSIFIER)
    parameter_range = script_args.get('PARAMETER_RANGE') # should be a dict with one key and value as a range

    ## optional script parameters
    sample_ids = script_args.get('SAMPLE_IDS',  default_val(exp_params, 'SAMPLE_IDS')),
    feature_ids = script_args.get('FEATURE_IDS', default_val(exp_params, 'FEATURE_IDS'))
    classifier_name = script_args.get('CLASSIFIER_NAME', default_val(exp_params, 'CLASSIFIER_NAME'))
    classifier_fweights_handle = script_args.get('CLASSIFIER_FWEIGHTS_HANDLE',
                                                 default_val(exp_params, 'CLASSIFIER_FWEIGHTS_HANDLE'))
    classifier_sweights_handle = script_args.get('CLASSIFIER_SWEIGHTS_HANDLE',
                                                 default_val(exp_params, 'CLASSIFIER_SWEIGHTS_HANDLE'))
    jump_ratio = script_args.get('JUMP_RATIO', 5)
    classifier_sorting_func = script_args.get('CLASSIFIER_SORTING_FUNC', lambda x: -np.abs(x))
    plot_args = script_args.get('PLOT_ARGS', dict()) # args for plotting, see https://matplotlib.org/3.2.2/api/axes_api.html
    figsize = script_args.get('FIGSIZE', (14, 10))
    save_name = script_args.get('SAVE_NAME', None)

    # intitialize output
    values = next(iter(parameter_range.values()))
    key = next(iter(parameter_range.keys()))
    num_features = []
    for val in values:

        # set the attribute on the classifier
        setattr(classifier, key, val)

        # classify
        classification_results = ds.classify(classifier=classifier,
                                             classifier_name=classifier_name,
                                             attr=class_attr,
                                             sample_ids=sample_ids,
                                             feature_ids=feature_ids,
                                             scorer=bsr,
                                             scorer_name='bsr',
                                             f_weights_handle=classifier_fweights_handle,
                                             s_weights_handle=classifier_sweights_handle)

        # gather # important features in model
        if classifier_fweights_handle is not None:
            weights = classification_results['f_weights'].filter(regex='weights').squeeze()
        elif classifier_sweights_handle is not None:
            weights = classification_results['s_weights'].filter(regex='weights').squeeze()
        weights = weights[~weights.isna()]
        weights = weights.apply(classifier_sorting_func)
        ordering = weights.argsort().values
        weights = weights.iloc[ordering]
        a = weights.iloc[:-1].values
        b = weights.iloc[1:].values
        ratios = np.divide(a, b, out=np.ones_like(a)*np.inf, where=b != 0)
        try:
            id = np.where(ratios > jump_ratio)[0][0] + 1
        except IndexError:
            id = len(a)
            print("Jump failed, no features selected. Try a different jump ratio.")

        # append result to list
        num_features.append(id/len(a))
        pass

    # transform result to ndarray
    num_features = np.array(num_features)

    # plot the results
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.plot(values,
            num_features,
            #label='Number of features',

            )
    ax.update(dict(xlabel=key,
                   ylabel='Proportion of Features'))
    ax.update(plot_args)
    if save_name is None:
        save_name = '_'.join([ds.name, exp_name, classifier_name, class_attr.lower(), key, 'feature_counts'])
    plt.savefig(fname=os.path.join(fig_dir, save_name + '.png'), format='png')
    plt.show()




