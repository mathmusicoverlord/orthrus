"""
Generic script for running a pipeline on a dataset.
"""


# imports for arguments
import argparse
import os

# command line arguments
parser = argparse.ArgumentParser("generic-pipeline-processor")

parser.add_argument('--exp_params',
                    type=str,
                    default=os.path.join(os.environ['ORTHRUS_PATH'], 'test_data', 'Iris', 'Experiments',
                                         'classify_setosa_versicolor_svm',
                                         'classify_setosa_versicolor_svm_params.py'),
                    help='File path of containing the experimental parameters. Default is the Iris experiment.')

parser.add_argument('--pipeline',
                    type=str,
                    help='File path or file name of containing the pipeline. If just a file name is provided '
                         'then the file is assumed to live in the Pipelines subdirectory of the experiment directory '
                         'where the experimental parameters reside.')

parser.add_argument('--checkpoint',
                    type=bool,
                    default=False,
                    help='Flag indicating whether or not to save checkpoints of pipeline. See Pipeline docs.')

parser.add_argument('--stop_before',
                    default=None,
                    help='Integer or string indicating where to stop the pipeline. See Pipeline docs.')


args = parser.parse_args()

# imports
from orthrus.core.helper import save_object, module_from_path, default_val, pop_first_element as pop
from orthrus.core.dataset import DataSet
from orthrus.core.pipeline import Pipeline
from orthrus.core.helper import generate_save_path, load_object
import inspect
from typing import Union

# set args
checkpoint = args.checkpoint
stop_before = args.stop_before
if stop_before is not None and stop_before.isnumeric():
    stop_before = int(stop_before)

# set experiment parameters
exp_params = module_from_path('exp_params', args.exp_params)


# required script params
results_dir = exp_params.RESULTS_DIR
exp_dir = exp_params.EXP_DIR
exp_name = exp_params.EXP_NAME
ds = exp_params.DATASET

# optional script params
sample_ids = default_val(exp_params, 'SAMPLE_IDS')
feature_ids = default_val(exp_params, 'FEATURE_IDS')

# find pipeline
pipeline_path = args.pipeline
split_path  = os.path.split(pipeline_path)
if split_path[0] == '':
    if '.py' not in split_path[1]:
        pipeline_path = pipeline_path + '.py'
    pipeline_path = os.path.join(exp_dir, 'Pipelines', pipeline_path)
else:
    pipeline_path = os.path.abspath(pipeline_path)

# set pipeline object
pipeline_module = module_from_path('pipeline_module', pipeline_path)
pipeline_module_members = inspect.getmembers(pipeline_module)
pipeline = [obj for obj in pipeline_module_members if isinstance(obj[1], Pipeline)]
if len(pipeline) == 0:
    raise NameError("No Pipeline instance exists in %s!" % (pipeline_path,))
elif len(pipeline) > 1:
    raise ValueError("Too many Pipeline instances in %s. Please provide one Pipeline instance per file!" % (pipeline_path,))
else:
    pipeline = pipeline[0][1]

# generate default checkpoint path
default_checkpoint_path = os.path.join(results_dir,
                                       pipeline.pipeline_name,
                                       pipeline.pipeline_name + '.pickle')

# gather Pipeline parameters
pipeline_params = list(inspect.signature(Pipeline).parameters.keys())
pipeline_params.remove('checkpoint_path')
pipeline_params.append('_checkpoint_path')

# assign default checkpoint path
if pipeline.checkpoint_path is None:
    os.makedirs(os.path.split(default_checkpoint_path)[0], exist_ok=True)  # make the directory
    pipeline.checkpoint_path = default_checkpoint_path

# define the script run function
def run(ds: DataSet,
        pipeline: Pipeline,
        checkpoint: bool,
        stop_before: Union[int, str]=None,
        sample_ids=None,
        feature_ids=None,
        ):

# assign the default path

    # check for checkpointing
    if checkpoint:
        # create copy of pipepline with correct checkpoint path
        pipeline = Pipeline(**{k.lstrip('_'): pipeline.__dict__[k] for k in pipeline_params})

    # slice dataset
    ds_new = ds.slice_dataset(sample_ids=sample_ids, feature_ids=feature_ids)

    # classify data
    pipeline.run(ds_new,
                 checkpoint=checkpoint,
                 stop_before=stop_before)

    return pipeline

# define the script save function
def save(pipeline: Pipeline, checkpoint: bool):

        if not checkpoint:
            # save pipeline
            pipeline.save(pipeline.checkpoint_path, overwrite=True)


if __name__ == '__main__':

    # run the script
    pipeline = run(ds=pop(ds),
                   pipeline=pop(pipeline),
                   checkpoint=pop(checkpoint),
                   stop_before=pop(stop_before),
                   sample_ids=pop(sample_ids),
                   feature_ids=pop(feature_ids),
                   )

    # save the results
    save(pipeline, checkpoint=pop(checkpoint))