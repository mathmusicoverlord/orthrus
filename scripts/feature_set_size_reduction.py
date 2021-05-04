"""
Generic script for running feature set size reduction on a dataset.
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
        from datasci.core.helper import save_object, load_object
        from datasci.core.helper import module_from_path
        import datasci.sparse.feature_selection.helper as fhelper

        # set experiment parameters
        exp_params = module_from_path('exp_params', args.exp_params)
        results_dir = exp_params.RESULTS_DIR
        exp_name = exp_params.EXP_NAME
        class_attr = exp_params.CLASS_ATTR
        ds = exp_params.DATASET
        sample_ids = exp_params.SAMPLE_IDS
        feauture_selection_results_file_name = exp_params.FEATURE_SELECTION_RESULTS_FILE_NAME
        feature_set_size_reduction_file_name = exp_params.FEATURE_SET_SIZE_REDUCTION_RESULTS_FILE_NAME

        model_factory = exp_params.MODEL_FACTORY
        score = exp_params.SCORE
        ranking_method = exp_params.RANKING_METHOD
        ranking_method_args = exp_params.RANKING_METHOD_ARGS
        partitoner = exp_params.PARTITIONER
        start = exp_params.START
        end = exp_params.END
        jump = exp_params.JUMP   
        feature_selection_result = load_object(file_path=os.path.join(results_dir, feauture_selection_results_file_name))
        features = feature_selection_result['f_results']
        try:
                sample_ids_validation = exp_params.SAMPLE_IDS_VALIDATION
        except AttributeError as e:
                print('sample_ids_validation not defined in the params file. Using sample_ids_validation = sample_ids instead.')
                sample_ids_validation = sample_ids

        reduced_feature_results = fhelper.reduce_feature_set_size(ds, 
                                features, 
                                sample_ids,
                                class_attr,
                                model_factory, 
                                score, 
                                ranking_method,
                                ranking_method_args,
                                test_sample_ids=sample_ids_validation,
                                #partitioner=partitioner,
                                start = start, 
                                end = end, 
                                jump = jump)
        save_object(reduced_feature_results, file_path=os.path.join(results_dir, feature_set_size_reduction_file_name))

        # import sklearn
        # import datasci.sparse.feature_selection.helper as fhelper
        # partitioner = sklearn.model_selection.LeaveOneGroupOut()

        # fhelper.rank_features_within_attribute_class(features, 
        #                                 'frequency', 
        #                                 'freq_weights',
        #                                 ds, 
        #                                 partitioner, 
        #                                 sample_ids,
        #                                 score,
        #                                 class_attr,
        #                                 model_factory,
        #                                 f_weights_handle = 'coef_',
        #                                 feature_ids = features['frequency'] >= 230)
        # feature_selection_result['f_results'] = features
        # save_object(feature_selection_result, os.path.join(results_dir, feauture_selection_results_file_name))
