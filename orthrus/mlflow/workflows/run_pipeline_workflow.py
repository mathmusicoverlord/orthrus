# imports
import argparse
import logging
import os
import mlflow
from urllib.parse import urlparse
from pathlib import Path

from orthrus.core.helper import module_from_path, reconstruct_logger_from_details, extract_reconstruction_details_from_logger
from orthrus.core.pipeline import Pipeline, Report

import orthrus.mlflow.modules.utils as utils

logger = logging.getLogger(__name__)

# user defined functions
def set_description():
    mlflow.set_tag('mlflow.note.content','This is a value')

def run(experiment_module, **kwargs):


    #get environment variables
    experiment_variables : dict = experiment_module.get_experiment_variables()

    # load and slice dataset
    
    ds = utils.slice_dataset(experiment_variables)


    #remove ds before logging the params
    experiment_variables.pop('ds')

    #log experiment variables
    mlflow.log_params(experiment_variables)


    # run the pipeline on the data
    artifacts_dir = urlparse(mlflow.get_artifact_uri()).path
    iter_num = kwargs.get("iter", None)
    if iter_num is None:
        run_dir = artifacts_dir
    else:
        run_dir = os.path.join(artifacts_dir, f'run_{iter_num}')
        os.makedirs(run_dir, exist_ok=True)

    filename = kwargs.get('workflow_name', None)
    if filename is None:
        filename = f'{experiment_module.__name__}'
        
    filename = f'{filename}_{kwargs.get("iter", "0")}_.pickle'

    checkpoint_path = os.path.join(run_dir, filename)

    # get pipeline
    kwargs['checkpoint_path']=checkpoint_path    
    pipeline : Pipeline = experiment_module.generate_pipeline(**kwargs)

    utils.log_pipeline_processes(pipeline)
    pipeline.run(ds, checkpoint=kwargs.get('checkpoint', True))

    args = {'results_location': run_dir,
            'pipeline': pipeline}


    try:
        experiment_module.process_results(**args)
    except AttributeError as e:
        logger.error(e, exc_info=True)
        # logger.error('The experiment module does not contain "process_results" method.')
    except:
        logger.error(e, exc_info=True)    

    # set description
    set_description()

    return pipeline, pipeline.results_
        

if __name__=="__main__":

    parser = utils.get_workflow_parser()
    args = utils.process_args(parser)

    experiment_module = module_from_path(Path(args.experiment_module_path).stem, args.experiment_module_path)
    
    args = vars(args)
    logger = utils.setup_workflow_execution(args, log_filename = 'execution.log')
    args.pop('experiment_module_path')

    experiment_workflow_args = experiment_module.get_pipeline_workflow_args(args)

    run(experiment_module, **experiment_workflow_args)