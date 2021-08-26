"""
Generic script for computing batch correction metric on a dataset.
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
from datasci.core.helper import save_object, module_from_path, default_val, pop_first_element as pop

# set experiment parameters
exp_params = module_from_path('exp_params', args.exp_params)
try:
    script_args = exp_params.BATCH_CORRECTION_ARGS
except AttributeError:
    script_args = {}
        
## required script params
results_dir = script_args.get('RESULTS_DIR', exp_params.RESULTS_DIR)
exp_name = script_args.get('EXP_NAME', exp_params.EXP_NAME)
attr = script_args.get('CLASS_ATTR', exp_params.CLASS_ATTR)
ds = script_args.get('DATASET', exp_params.DATASET)

## optional script params
sample_ids = script_args.get('SAMPLE_IDS', default_val(exp_params, 'SAMPLE_IDS'))
attribute = script_args.get('CLASS_ATTR', default_val(exp_params, 'CLASS_ATTR'))

features_file = script_args.get('FEATURE_SELECTION_RESULTS_FILE_NAME', default_val(exp_params, 'FEATURE_SELECTION_RESULTS_FILE_NAME'))

ranking_method_handle = script_args.get('RANKING_METHOD', default_val(exp_params, 'RANKING_METHOD'))
ranking_method_args = script_args.get('RANKING_METHOD_ARGS', default_val(exp_params, 'RANKING_METHOD_ARGS'))


batch_correction_metric_handle = script_args.get('BATCH_CORRECTION_METRIC_HANDLE', default_val(exp_params, 'BATCH_CORRECTION_METRIC_HANDLE'))
batch_correction_metric_args = script_args.get('BATCH_CORRECTION_METRIC_ARGS', default_val(exp_params, 'BATCH_CORRECTION_METRIC_ARGS'))
results_file_name = script_args.get('BATCH_CORRECTION_RESULTS_FILE_NAME', default_val(exp_params, 'BATCH_CORRECTION_RESULTS_FILE_NAME'))
num_cpus_per_worker = script_args.get('NUM_CPUS_PER_WORKER', default_val(exp_params, 'NUM_CPUS_PER_WORKER', 1))
num_gpus_per_worker = script_args.get('NUM_GPUS_PER_WORKER', default_val(exp_params, 'NUM_GPUS_PER_WORKER', 0))

local_mode = script_args.get('LOCAL_MODE', default_val(exp_params, 'LOCAL_MODE', False))

from datasci.sparse.feature_selection.helper import get_batch_correction_matric_for_ranked_features

# define the script run function
def run(ds,
        features_file,
        attr,
        ranking_method_handle,
        ranking_method_args,
        batch_correction_metric_handle,
        batch_correction_metric_args,
        sample_ids,
        num_cpus_per_worker,
        num_gpus_per_worker,
        local_mode,
        ):

        from datasci.core.helper import load_object
        features_dataframe = load_object(os.path.join(results_dir, features_file))['f_results']
        # classify data
        result = get_batch_correction_matric_for_ranked_features(ds, 
                        features_dataframe, 
                        attr,
                        ranking_method_handle,
                        ranking_method_args,
                        batch_correction_metric_handle,
                        batch_correction_metric_args,
                        sample_ids,
                        verbose_frequency=100,
                        num_cpus_per_worker=num_cpus_per_worker,
                        num_gpus_per_worker=num_gpus_per_worker,
                        local_mode=local_mode
                        )

        return result

# define the script save function
def save(result, num=None):
    # check for non-parallel job
    if num is None:
        num = ''

    # save classification results
    if results_file_name is None:
        save_object(result,
                    os.path.join(results_dir,
                                 '_'.join([ds.name,
                                           exp_name,
                                           'batch_correction_results',
                                           str(num) + '.pickle']
                                          )
                                 )
                    )
    else:
        save_object(result,
                    os.path.join(results_dir, '_'.join([results_file_name, str(num) + '.pickle']))
                    )


if __name__ == '__main__':

    # run the script
    batch_correction_results = run(pop(ds),
                                    pop(features_file),
                                    pop(attr),
                                    pop(ranking_method_handle),
                                    pop(ranking_method_args),
                                    pop(batch_correction_metric_handle),
                                    pop(batch_correction_metric_args),
                                    pop(sample_ids),
                                    pop(num_cpus_per_worker),
                                    pop(num_gpus_per_worker),
                                    pop(local_mode))

    # save the results
    save(batch_correction_results)
