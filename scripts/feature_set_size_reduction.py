"""
Generic script for running feature set size reduction on a dataset.
"""

# imports for arguments
import argparse
import os
from pickle import NONE
from datasci.core.helper import load_object
from datasci.sparse.feature_selection.helper import reduce_feature_set_size
# command line arguments
parser = argparse.ArgumentParser("generic-ifr")

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
    script_args = exp_params.FEATURE_SET_SIZE_REDUCTION_ARGS
except AttributeError:
    script_args = {}

## required script params
results_dir = script_args.get('RESULTS_DIR', exp_params.RESULTS_DIR)
exp_name = script_args.get('EXP_NAME', exp_params.EXP_NAME)
attr = script_args.get('CLASS_ATTR', exp_params.CLASS_ATTR)
ds = script_args.get('DATASET', exp_params.DATASET)

sample_ids = script_args.get('SAMPLE_IDS', default_val(exp_params, 'SAMPLE_IDS'))
feauture_selection_results_file_name = script_args.get('FEATURE_SELECTION_RESULTS_FILE_NAME',
                                    default_val(exp_params, 'FEATURE_SELECTION_RESULTS_FILE_NAME'))
results_file_name = script_args.get('FEATURE_SET_SIZE_REDUCTION_RESULTS_FILE_NAME',
                                    default_val(exp_params, 'FEATURE_SET_SIZE_REDUCTION_RESULTS_FILE_NAME'))

model = script_args.get('MODEL', default_val(exp_params, 'MODEL'))
score = script_args.get('SCORE', default_val(exp_params, 'SCORE'))
ranking_method = script_args.get('RANKING_METHOD', default_val(exp_params, 'RANKING_METHOD'))
ranking_method_args = script_args.get('RANKING_METHOD_ARGS', default_val(exp_params, 'RANKING_METHOD_ARGS'))

partitioner = script_args.get('PARTITIONER', default_val(exp_params, 'PARTITIONER'))
start = script_args.get('START', default_val(exp_params, 'START'))
end = script_args.get('END', default_val(exp_params, 'END'))
step = script_args.get('STEP', default_val(exp_params, 'STEP'))

sample_ids_validation = script_args.get('SAMPLE_IDS_VALIDATION', default_val(exp_params, 'SAMPLE_IDS_VALIDATION', None))

num_cpus_per_worker = script_args.get('NUM_CPUS_PER_WORKER', default_val(exp_params, 'NUM_CPUS_PER_WORKER', 1))
num_gpus_per_worker = script_args.get('NUM_GPUS_PER_WORKER', default_val(exp_params, 'NUM_GPUS_PER_WORKER', 0))

if sample_ids_validation is None:
    print('sample_ids_validation not defined in the params file. Using sample_ids_validation = sample_ids instead.')
    sample_ids_validation = sample_ids


def run(ds, 
        feauture_selection_results_file_name, 
        sample_ids,
        attr,
        model, 
        score, 
        ranking_method,
        ranking_method_args,
        sample_ids_validation,
        partitioner,
        start, 
        end, 
        step,
        num_cpus_per_worker,
        num_gpus_per_worker):

    feature_selection_result = load_object(file_path=os.path.join(results_dir, feauture_selection_results_file_name))
    features = feature_selection_result['f_results']

    reduced_feature_results = reduce_feature_set_size(ds, 
                            features, 
                            sample_ids,
                            attr,
                            model, 
                            score, 
                            ranking_method,
                            ranking_method_args,
                            test_sample_ids=sample_ids_validation,
                            partitioner=partitioner,
                            start=start, 
                            end=end, 
                            step=step,
                            num_cpus_per_worker=num_cpus_per_worker,
                            num_gpus_per_worker=num_gpus_per_worker)
    
    return reduced_feature_results

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
                                           'feature_set_size_reduction',
                                           'results',
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
    feature_set_size_reduciton_results = run(pop(ds),
                                            pop(feauture_selection_results_file_name), 
                                            pop(sample_ids),
                                            pop(attr),
                                            pop(model), 
                                            pop(score), 
                                            pop(ranking_method),
                                            pop(ranking_method_args),
                                            pop(sample_ids_validation),
                                            pop(partitioner),
                                            pop(start), 
                                            pop(end), 
                                            pop(step),
                                            pop(num_cpus_per_worker),
                                            pop(num_gpus_per_worker)
                                            )
                            
    # save the results
    save(feature_set_size_reduciton_results)