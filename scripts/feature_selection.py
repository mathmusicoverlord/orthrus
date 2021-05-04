"""
Generic script for running feature selection on a dataset.
"""

if __name__ == '__main__':
        # imports for arguments
        import argparse
        import os

        # command line arguments
        parser = argparse.ArgumentParser("generic-ifr")

        parser.add_argument('--exp_params',
                        type=str,
                        required=True,
                        help='File path of containing the experimental parameters.')

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
        feature_selector = exp_params.FEATURE_SELECTOR
        selector_name = exp_params.SELECTOR_NAME
        results_handle = exp_params.FEATURE_SELECTION_RESULTS_HANDLE
        results_file_name = exp_params.FEATURE_SELECTION_RESULTS_FILE_NAME


        result = ds.feature_select(feature_selector,
                        class_attr,
                        selector_name=selector_name,
                        sample_ids=sample_ids,
                        f_results_handle=results_handle,
                        append_to_meta=False,
                        )

        # save feature selection results
        save_object(result, os.path.join(results_dir, results_file_name))
