# imports
from urllib.parse import urlparse
import mlflow
import mlflow
from pathlib import Path
import sys
from mlflow.entities import experiment_tag
from orthrus.core.helper import module_from_path
import os
import orthrus.mlflow.modules.utils as utils



if __name__ == '__main__':

        # process command line arguments
        parser = utils.get_workflow_parser('Pipeline of WorkflowManagers', 'Runs a pre-defined defined orthrus pipeline', add_exp_module_path=True)
        args = vars(utils.process_args(parser))

        # setup logger, mlflow and ray runs
        logger = utils.setup_workflow_execution(**args)
        
        # generate module
        experiment_name = Path(args['experiment_module_path']).stem
        experiment_module = module_from_path(experiment_name, args['experiment_module_path'])

        #get pipeline of workflows
        pipeline = experiment_module.generate_pipeline_of_workflows(**args)



