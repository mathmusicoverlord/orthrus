"""
Generic script for classifying a dataset.
"""


# imports for arguments
import argparse
import os

# command line arguments
parser = argparse.ArgumentParser("generic-classification")

parser.add_argument('--exp_params',
                    type=str,
                    default=os.path.join(os.environ['DATASCI_PATH'], 'test_data', 'Iris', 'Experiments',
                                         'setosa_versicolor_classify_species_svm',
                                         'classify_setosa_versicolor_svm_params.py'),
                    help='File path of containing the experimental parameters. Default is the Iris experiment.')

parser.add_argument('--score',
                    type=str,
                    default='bsr',
                    choices=['bsr', 'accuracy', 'confusion'],
                    help='Score reported on the prediction results.')

args = parser.parse_args()

# imports
from datasci.core.helper import save_object, module_from_path, default_val, pop_first_element as pop

# set experiment parameters
exp_params = module_from_path('exp_params', args.exp_params)
try:
    script_args = exp_params.CLASSIFY_ARGS
except AttributeError:
    script_args = {}

## required script params
results_dir = script_args.get('RESULTS_DIR', exp_params.RESULTS_DIR)
exp_name = script_args.get('EXP_NAME', exp_params.EXP_NAME)
attr = script_args.get('CLASS_ATTR', exp_params.CLASS_ATTR)
ds = script_args.get('DATASET', exp_params.DATASET)
classifier = script_args.get('CLASSIFIER', default_val(exp_params, 'CLASSIFIER'))

## optional script params
sample_ids = script_args.get('SAMPLE_IDS',  default_val(exp_params, 'SAMPLE_IDS')),
feature_ids = script_args.get('FEATURE_IDS', default_val(exp_params, 'FEATURE_IDS'))
classifier_name = script_args.get('CLASSIFIER_NAME', default_val(exp_params, 'CLASSIFIER_NAME'))
scorer_args = script_args.get('SCORER_ARGS', default_val(exp_params, 'CLASSIFY_SCORER_ARGS', {}))
f_weights_handle = script_args.get('CLASSIFIER_F_WEIGHTS_HANDLE', default_val(exp_params, 'CLASSIFIER_F_WEIGHTS_HANDLE'))
s_weights_handle = script_args.get('CLASSIFIER_S_WEIGHTS_HANDLE', default_val(exp_params, 'CLASSIFIER_S_WEIGHTS_HANDLE'))
partitioner = script_args.get('PARTITIONER', default_val(exp_params, 'PARTITIONER'))
partitioner_name = script_args.get('PARTITIONER_NAME', default_val(exp_params, 'PARTITIONER_NAME'))
results_file_name = script_args.get('RESULTS_FILE_NAME',
                                    default_val(exp_params, 'CLASSIFICATION_RESULTS_FILE_NAME'))
training_transform = script_args.get('TRAINING_TRANSFORM', default_val(exp_params, 'CLASSIFY_TRAINING_TRANSFORM'))

# set scorer
scorer_name = args.score
if args.score == 'bsr':
    from sklearn.metrics import balanced_accuracy_score
    scorer = balanced_accuracy_score
elif args.score == 'confusion':
    from sklearn.metrics import confusion_matrix
    scorer = confusion_matrix
elif args.score == 'accuracy':
    from sklearn.metrics import accuracy_score
    scorer = accuracy_score

# define the script run function
def run(ds,
        classifier,
        classifier_name,
        attr,
        sample_ids,
        feature_ids,
        partitioner,
        partitioner_name,
        scorer,
        scorer_name,
        scorer_args,
        f_weights_handle,
        s_weights_handle,
        training_transform,
        ):

    # classify data
    classification_results = ds.classify(classifier=classifier,
                                         classifier_name=classifier_name,
                                         attr=attr,
                                         sample_ids=sample_ids,
                                         feature_ids=feature_ids,
                                         partitioner=partitioner,
                                         partitioner_name=partitioner_name,
                                         scorer=scorer,
                                         scorer_name=scorer_name,
                                         scorer_args=scorer_args,
                                         f_weights_handle=f_weights_handle,
                                         s_weights_handle=s_weights_handle,
                                         training_transform=training_transform,
                                         )

    return classification_results

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
                                           args.score,
                                           'classification',
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
    classification_results = run(ds=pop(ds),
                                 classifier=pop(classifier),
                                 classifier_name=pop(classifier_name),
                                 attr=pop(attr),
                                 sample_ids=pop(sample_ids),
                                 feature_ids=pop(feature_ids),
                                 partitioner=pop(partitioner),
                                 partitioner_name=pop(partitioner_name),
                                 scorer=pop(scorer),
                                 scorer_name=pop(scorer_name),
                                 scorer_args=pop(scorer_args),
                                 f_weights_handle=pop(f_weights_handle),
                                 s_weights_handle=pop(s_weights_handle),
                                 training_transform=pop(training_transform))
    # save the results
    save(classification_results)