"""
Generic script for running feature selection on a dataset.
"""

if __name__ == '__main__':
        # imports for arguments
        import argparse
        import os

        # command line arguments
        parser = argparse.ArgumentParser("generic-feature-selection")

        parser.add_argument('--exp_params',
                        type=str,
                        required=True,
                        help='File path of containing the experimental parameters.')

        args = parser.parse_args()

        # imports
        from sklearn.metrics import balanced_accuracy_score as bsr
        from datasci.core.helper import save_object
        from datasci.core.helper import module_from_path
        from datasci.core.helper import default_val

        # set experiment parameters
        exp_params = module_from_path('exp_params', args.exp_params)
        script_args = exp_params.FEATURE_SELECTION_ARGS

        ## required script params
        results_dir = script_args.get('RESULTS_DIR', exp_params.RESULTS_DIR)
        exp_name = script_args.get('EXP_NAME', exp_params.EXP_NAME)
        class_attr = script_args.get('CLASS_ATTR', exp_params.CLASS_ATTR)
        ds = script_args.get('DATASET', exp_params.DATASET)
        classifier = script_args.get('CLASSIFIER', exp_params.CLASSIFIER)
        feature_selector = script_args.get('FEATURE_SELECTOR', exp_params.FEATURE_SELECTOR)
        results_handle = script_args.get('RESULTS_HANDLE',
                                         exp_params.FEATURE_SELECTION_RESULTS_HANDLE)

        ## optional script params
        sample_ids = script_args.get('SAMPLE_IDS', default_val(exp_params, 'SAMPLE_IDS')),
        feature_ids = script_args.get('FEATURE_IDS', default_val(exp_params, 'FEATURE_IDS'))
        selector_name = script_args.get('FEATURE_SELECTOR_NAME', default_val(exp_params, 'FEATURE_SELECTOR_NAME'))
        results_file_name = script_args.get('RESULTS_FILE_NAME',
                                            default_val(exp_params, 'FEATURE_SELECTION_RESULTS_FILE_NAME'))


        result = ds.feature_select(feature_selector,
                        class_attr,
                        selector_name=selector_name,
                        sample_ids=sample_ids,
                        feature_ids=feature_ids,
                        f_results_handle=results_handle,
                        append_to_meta=False,
                        )

        # save feature selection results
        if results_file_name is None:
            save_object(result,
                        os.path.join(results_dir,
                                     '_'.join([ds.name, exp_name, args.score, 'feature_selection_results.pickle'])))
        else:
            save_object(result, os.path.join(results_dir, results_file_name))
