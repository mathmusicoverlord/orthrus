import logging
logger = logging.getLogger(__name__)

def evaluate_ssvm_ifr_tune(result, **kwargs):
    last_iter = kwargs['n_iter']
    best_config_from_last_iteration = result[last_iter]['result']

    verdict = 'pass'
    params = {}

    # check if this was a tune run or a standalone run.
    # if it was a standalone run, there is no search space, so there is no search space to update
    tune_config = kwargs.get('tune_config', None)
    if tune_config is not None:
        # check ssvm C values
        ssvm_C_search_space = tune_config['ssvm_C']
        value_range = ssvm_C_search_space.upper - ssvm_C_search_space.lower

    
        if best_config_from_last_iteration['ssvm_C'] - ssvm_C_search_space.lower == 0:
            # move the search space to the left
            logger.warn(f'The best ssvm C parameter value ({best_config_from_last_iteration["ssvm_C"]}) was found by tuning ' \
                f'is very close to the lower end of search space [{ssvm_C_search_space.lower}, {ssvm_C_search_space.upper-1}]')
            ssvm_C_search_space.lower -= (value_range - int(.4 * value_range))
            ssvm_C_search_space.upper -= (value_range - int(.4 * value_range))
            logger.info(f'New ssvm_C search space [{ssvm_C_search_space.lower}, {ssvm_C_search_space.upper-1}]')

            verdict = 'warn'


        elif (ssvm_C_search_space.upper-1) - best_config_from_last_iteration['ssvm_C'] == 0 :
            # move the search space to the right
            logger.warn(f'The ssvm C parameter value ({best_config_from_last_iteration["ssvm_C"]}) was found by tuning ' \
                f'is very close to the upper end of search space [{ssvm_C_search_space.lower}, {ssvm_C_search_space.upper-1}]')
            ssvm_C_search_space.lower += (value_range - int(.4 * value_range))
            ssvm_C_search_space.upper += (value_range - int(.4 * value_range))
            logger.info(f'New ssvm_C search space [{ssvm_C_search_space.lower}, {ssvm_C_search_space.upper-1}]')

            verdict = 'warn'


        # check jump ratio values (for this parameter, we only increase the values)

        jumpratio_search_space = tune_config['ifr_jumpratio']
        try:
            value_range = jumpratio_search_space.upper - jumpratio_search_space.lower

            if best_config_from_last_iteration['ifr_jumpratio'] == jumpratio_search_space.upper-1:
                logger.warn(f'The ifr_jumpratio parameter value ({best_config_from_last_iteration["ifr_jumpratio"]}) was found by tuning ' \
                    f'is very close to the upper end of search space [{jumpratio_search_space.lower}, {jumpratio_search_space.upper-1}]')
                # move the search space to the left
                jumpratio_search_space.lower += (value_range - int(.4 * value_range))
                jumpratio_search_space.upper += (value_range - int(.4 * value_range))
                logger.info(f'New jumpratio search space [{jumpratio_search_space.lower}, {jumpratio_search_space.upper-1}]')

                verdict = 'warn'
        except AttributeError:
            # there is no search space to update
            pass

        params['tune_config'] = tune_config
        num_selected_features = best_config_from_last_iteration['all_results'].iloc[0]['total_features_extracted']

    else:
        num_selected_features = 0
        # extract mean bsr across folds
        for batch, results in best_config_from_last_iteration.items():
            ifr = results['selector']
            features_for_batch = ifr.diagnostic_information_['true_feature_count'].sum().sum()
            if features_for_batch == 0:
                num_selected_features = 0
                break
            num_selected_features += features_for_batch

    if num_selected_features == 0:
        logger.error('Feature selection for one or more batches did not extract any features.')
        verdict = 'fail'

    logger.info(f'Finishing up evaluation of the workflow. Evaluation result is "{verdict}".')

    return {'verdict': verdict,
            'params': params
            }


def evaluate_svm_c_tune(result, **kwargs):

    last_iter = kwargs['n_iter']
    best_config_from_last_iteration = result[last_iter]['result']

    verdict = 'pass'
    params = {}

    tune_config = kwargs['tune_config']
    
    # check ssv C values
    svm_C_search_space = tune_config['svm_C']
    value_range = svm_C_search_space.upper - svm_C_search_space.lower

    if best_config_from_last_iteration['svm_C'] - svm_C_search_space.lower == 0:
        # move the search space to the left
        logger.warn(f'The best svm C parameter value ({best_config_from_last_iteration["svm_C"]}) was found by tuning ' \
            f'is very close to the lower end of search space [{svm_C_search_space.lower}, {svm_C_search_space.upper-1}]')
        svm_C_search_space.lower -= (value_range - int(.4 * value_range))
        svm_C_search_space.upper -= (value_range - int(.4 * value_range))
        logger.info(f'New svm_C search space [{svm_C_search_space.lower}, {svm_C_search_space.upper-1}]')

        verdict = 'warn'


    elif (svm_C_search_space.upper-1) - best_config_from_last_iteration['svm_C'] == 0 :
        # move the search space to the right
        logger.warn(f'The svm C parameter value ({best_config_from_last_iteration["svm_C"]}) was found by tuning ' \
            f'is very close to the upper end of search space [{svm_C_search_space.lower}, {svm_C_search_space.upper-1}]')
        svm_C_search_space.lower += (value_range - int(.4 * value_range))
        svm_C_search_space.upper += (value_range - int(.4 * value_range))
        logger.info(f'New svm_C search space [{svm_C_search_space.lower}, {svm_C_search_space.upper-1}]')

        verdict = 'warn'

    params['tune_config'] = tune_config

    logger.info(f'Finishing up evaluation of the workflow. Evaluation result is "{verdict}".')

    return {'verdict': verdict,
            'params': params
            }




# def compute_average(series):
#         total_sum = series.sum().sum()
#         non_nan_values = (~series.isna()).sum().sum()
#         return total_sum / non_nan_values

# def evaluate_SSVM_IFR(result, **kwargs):

#     # for now let's just work with the most recent result, and not worry about trends
#     last_iter = kwargs['iter']
#     latest_result = result[last_iter]['result']

#     evaluation_verdict = []
#     evaluation_message = []
#     recommended_param_updates = {}
#     for batch_id, batch_result in latest_result.items():

#         ifr = batch_result['selector']
#         diag_info = ifr.diagnostic_information_

#         stopping_conditions = diag_info['stopping_conditions']
#         # stopping_conditions.set_index('Stopping Conditions', inplace=True)

#         feature_counts = diag_info['true_feature_count']
#         training_samples = diag_info['num_training_samples']

#         ssvm_C = ifr.classifier.C



#         # ------------------------------------
#         # condition to check if ifr selected appropriate number of features
#         # ------------------------------------
        
#         # check 1: if too many features
#         if 'More features than cutoff' in stopping_conditions.index:
#             ratio = stopping_conditions.loc['More features than cutoff']['Frequency'] / stopping_conditions['Frequency'].sum()

#             if ratio > kwargs.get('max_feature_cutoff_ratio', 0.05):
#                 evaluation_message.append(f'{batch_id}: "More feature selected than cutoff" test failed')
#                 evaluation_verdict.append('fail')
#                 recommended_param_updates['ssvm_C'] = ssvm_C / 2


#         # check 2 : if too few features
#         average_selected_feature_count = compute_average(feature_counts)
#         average_training_sample_count = compute_average(training_samples)

#         cutoff_for_ratio_of_avg_training_sample_count_to_avg_features_selected_count = kwargs.get('min_feature_cutoff_ratio', 1.5)
#         ratio_of_avg_training_sample_count_to_avg_features_selected_count = average_training_sample_count / average_selected_feature_count

#         if ratio_of_avg_training_sample_count_to_avg_features_selected_count > cutoff_for_ratio_of_avg_training_sample_count_to_avg_features_selected_count:
#             evaluation_message.append(f'{batch_id}: Ratio of average number of training samples to averane number of features selected \
#                         ({ratio_of_avg_training_sample_count_to_avg_features_selected_count}) is greater than {cutoff_for_ratio_of_avg_training_sample_count_to_avg_features_selected_count}')
#             evaluation_verdict.append('warn')
#             # recommended_param_updates['ssvm_C'] = ssvm_C * 2


#     evaluation_verdict = np.array(evaluation_verdict)
#     verdict = 'pass'
#     if np.any(evaluation_verdict == 'fail'):
#         verdict = 'fail'
#     elif np.any(evaluation_verdict == 'warn'):
#         verdict = 'warn'
    

#     return {'verdict': verdict,
#             'message': '\n'.join(evaluation_message),
#             'params': recommended_param_updates}
