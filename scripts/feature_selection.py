"""
Generic script for running feature selection on a dataset.
"""


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
from orthrus.core.helper import save_object, module_from_path, default_val, pop_first_element as pop

# set experiment parameters
exp_params = module_from_path('exp_params', args.exp_params)
try:
    script_args = exp_params.FEATURE_SELECTION_ARGS
except AttributeError:
    script_args = {}

## required script params
results_dir = script_args.get('RESULTS_DIR', exp_params.RESULTS_DIR)
exp_name = script_args.get('EXP_NAME', exp_params.EXP_NAME)
attr = script_args.get('CLASS_ATTR', exp_params.CLASS_ATTR)
ds = script_args.get('DATASET', exp_params.DATASET)
feature_selector = script_args.get('FEATURE_SELECTOR',
                                    default_val(exp_params, 'FEATURE_SELECTOR'))
f_results_handle = script_args.get('RESULTS_HANDLE',
                                    default_val(exp_params, 'FEATURE_SELECTION_RESULTS_HANDLE'))

## optional script params
sample_ids = script_args.get('SAMPLE_IDS', default_val(exp_params, 'SAMPLE_IDS'))
feature_ids = script_args.get('FEATURE_IDS', default_val(exp_params, 'FEATURE_IDS'))
feature_selector_name = script_args.get('FEATURE_SELECTOR_NAME', default_val(exp_params, 'FEATURE_SELECTOR_NAME'))
results_file_name = script_args.get('RESULTS_FILE_NAME',
                                    default_val(exp_params, 'FEATURE_SELECTION_RESULTS_FILE_NAME'))

save_selector = script_args.get('SAVE_SELECTOR',
                                    default_val(exp_params, 'SAVE_SELECTOR'))

append_to_meta = script_args.get('APPEND_TO_META',
                                    default_val(exp_params, 'APPEND_TO_META', False))

training_transform = script_args.get('TRAINING_TRANSFORM',
                                    default_val(exp_params, 'TRAINING_TRANSFORM', False))

# define the script run function
def run(ds,
        feature_selector,
        attr,
        feature_selector_name,
        sample_ids,
        feature_ids,
        f_results_handle,
        append_to_meta,
        training_transform):

    # classify data
    feature_selection_result = ds.feature_select(feature_selector,
                    attr,
                    selector_name=feature_selector_name,
                    sample_ids=sample_ids,
                    feature_ids=feature_ids,
                    f_results_handle=f_results_handle,
                    append_to_meta=append_to_meta,
                    training_transform=training_transform
                    )

    return feature_selection_result


# define the script save function
def save(result, num=None):
    # check for non-parallel job
    if num is None:
        num = ''

    # save classification results
    if results_file_name is None:
        path = os.path.join(results_dir,
                            '_'.join([ds.name,
                                    exp_name,
                                    'feature_selection_results',
                                    feature_selector_name,
                                    str(num)]
                                    )
                            )
    else:
        path = os.path.join(results_dir, '_'.join([results_file_name, str(num)]))

    #save features as csv                    
    result['f_results'].to_csv(path + '.csv')
    
    #save features as object
    if save_selector:
        save_object(result,
                    path + '.pickle'
                    )
    else:
        save_object({'f_results':result['f_results']},
            path + '.pickle'
            )

if __name__ == '__main__':

    # run the script
    feature_selection_results = run(pop(ds),
                                pop(feature_selector),
                                pop(attr),
                                feature_selector_name=pop(feature_selector_name),
                                sample_ids=pop(sample_ids),
                                feature_ids=pop(feature_ids),
                                f_results_handle=pop(f_results_handle),
                                append_to_meta=pop(append_to_meta),
                                training_transform = pop(training_transform))
                            
    # save the results
    save(feature_selection_results)
