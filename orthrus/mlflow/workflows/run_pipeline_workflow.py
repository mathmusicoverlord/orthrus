# imports
import logging
import mlflow
from orthrus.core.helper import module_from_path
from orthrus.core.pipeline import Pipeline
from orthrus.core.dataset import DataSet
import orthrus.mlflow.modules.utils as utils
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def set_description(description):
    mlflow.set_tag('mlflow.note.content', description)

def run(experiment_module_path:str, 
        ds: DataSet,
        sample_ids = None,
        feature_ids = None,
        checkpoint: bool=False,
        experiment_description: str=None,
        workflow_process_results_handle=None,
        workflow_name: str=None, 
        n_iter=None, 
        **kwargs):

    '''
    This :py:meth::`run` method does all the work to execute an experiment workflow by creating, running, and processing results of a pipeline. It can be used in two ways: 
    
        Option 1. Use :py:class::`orthrus.core.pipeline.WorkflowManager` process -> The WorkflowManager process calls the :py:meth::`run` method.

        Option 2. Run this script directly -> The :py:meth::`main` method of this script calls this :py:meth::`run` method.

    So, this :py:meth::`run` method is NOT designed to be called directly and therefore it is recommended to be used only using one of the two options listed above. 
    
    For a standalone usage, find an in-depth tutorial at ****************, and for use with :py:class::`orthrus.core.pipeline.WorkflowManager`, find the tutorial at *************.
    
    In either case, because this method is not called directly by users, the arguments to this method are provided indirectly. There are three methods in which users can do so, and for 
    some of these arguments, as you'll see below, the method may be predetermined or might change depending on if you're using option 1 or 2 from above. These methods are:

        - Using the command-line arguments.

        - Using the :py:meth::`get_workflow_args` method of the :py:attr::`experiment_module_path` script. 

        - Using :py:class::`orthrus.core.pipeline.WorkflowManager`'s attributes (only when using option 1)

    Additionally, you may note, as you continue reading, that some arguments to this run method like :py:attr::`n_iter` are set by :py:class::`orthrus.core.pipeline.WorkflowManager` when using option 1; users 
        don't need to worry about these. 
        
    A description of the arguments to this method and various options on how to provide them is listed below.

    Args:
        :py:attr::`experiment_module_path` (str): Path of the python module (Check the description of the main method to learn more about the requirements for the script). There are two ways to provide this 
                depending on whether the :py:meth::`run` method is called using option 1 or 2 from above. 

                For option 1: NOT to be set explicitly by the user. The value of :py:attr::`experiment_module_path` attribute of the :py:class::`orthrus.core.pipeline.WorkflowManager` is used.

                For option 2: provide using cmd-line args.


        :py:attr::`ds` (orthrus.core.dataset.Dataset):  (provide using :py:meth::`get_workflow_args` method) : Dataset object.  

        :py:attr::`sample_ids`: (provide using using cmd-line args or :py:meth::`get_workflow_args` method): The experiment will be restricted to just these samples from the ds. The value must be compatible with :py:meth::`orthrus.core.dataset.DataSet.slice_dataset` method.
        
        :py:attr::`feature_ids`: (provide using using cmd-line args or :py:meth::`get_workflow_args` method): The experiment will be restricted to just these features from the ds. The value must be compatible with :py:meth::`orthrus.core.dataset.DataSet.slice_dataset` method.
        
        :py:attr::`checkpoint` (bool): (provide using cmd-line args or :py:meth::`get_workflow_args` method) : Whether to save the pipeline checkpoint to disk or not.

        :py:attr::`experiment_description` (str): (provide using cmd-line args or :py:meth::`get_workflow_args` method) : Description of the experiment for mlflow tracking.

        :py:attr::`workflow_process_results_handle` : A handle to a method that processes the results of the pipeline after it has finished its execution. For instance, this method could save csv files and/or generate plots 
                and html files etc. There are two ways to provide this depending on whether the method is called using option 1 or 2 from above. Defaults to None, which means no processing of results is needed.

                for option 1: NOT to be set explicitly by the user. The value of :py:attr::`workflow_process_results_handle` attribute of the :py:class::`orthrus.core.pipeline.WorkflowManager` is used.

                for option 2: provide using :py:meth::`get_workflow_args` method.
            
                This method must take the following arguments:

                Args:
                    :py:attr::`results_location`: path of working directory where the results can be stored. defaults to the artifacts directory of the current mlflow run.
                    :py:attr::`pipeline`: a :py:class::`orthrus.core.pipeline.Pipeline` object which has finished its execution.
        
        :py:attr::`workflow_name` (str): Name of the pipeline checkpoint file. There are two ways to provide this depending on whether the method is called using option 1 or 2 from above.

                for option 1: (:py:class::`orthrus.core.pipeline.WorkflowManager`): NOT to be set explicitly by the user. The value of :py:attr::`name` attribute of the :py:class::`orthrus.core.pipeline.WorkflowManager` is used.

                for option 2: (cmd-line arg or :py:meth::`get_workflow_args`):  defaults to the name of the :py:attr::`experiment_module`. 

        :py:attr::`n_iter` (int): (:py:class::`orthrus.core.pipeline.WorkflowManager`): NOT to be set explicitly by the user.

    
    Returns:
        tuple of (finished :py:class::`orthrus.core.pipeline.Pipeline`, its results dictionary)
    '''

    #generate python module and get workflow arguments
    experiment_module = module_from_path(Path(experiment_module_path).stem, experiment_module_path)

    run_dir, filename, log_key = utils.get_results_dir_and_file_name(workflow_name, experiment_module, n_iter)
    
    # create checkpoint file path
    checkpoint_path = os.path.join(run_dir, filename)
    kwargs['checkpoint_path']=checkpoint_path  

    # add sample_ids and feature_ids to kwargs so they can be logged
    kwargs['sample_ids'] = sample_ids
    kwargs['feature_ids'] = feature_ids

    utils.reformat_kwargs_about_results_from_previous_process(kwargs, log_key)

    ds = ds.slice_dataset(sample_ids = sample_ids,
                            feature_ids = feature_ids)

    #log experiment variables
    # mlflow.log_params(experiment_variables)

    # get pipeline
    pipeline : Pipeline = experiment_module.generate_pipeline(**kwargs)

    #log parameters of the processes of the pipeline as artifacts
    utils.log_pipeline_processes(run_dir, pipeline)

    # run the pipeline on the data
    pipeline.run(ds, checkpoint=checkpoint)

    # process the results of the pipeline
    if workflow_process_results_handle is None:        
        workflow_process_results_handle(results_location = run_dir, results = pipeline)

    # set description
    if experiment_description is not None:
        set_description(experiment_description)

    return pipeline, pipeline.results_
        
def main():

    '''
    This script provides a general-purpose abstraction for creating, runing and processing the results of an orthrus pipeline. It requires a mandatory command-line argument - :py:attr::`experiment_module_path` - which specifies the 
    location of a python module. Please run "python /path/to/file/run_pipeline_workflow.py -h" to know about other command-line arguments you can use. The python module located at :py:attr::`experiment_module_path` must contain methods to get the 
    arguments of the experiment workflow and to generate the pipeline to be run. The exact specifications for these methods are detailed below:
    
        a. :py:meth::`get_workflow_args`.
            This method's purpose is to return a dictionary of arguments which may be used for running the workflow. The returned dictionary can contain infomation such as the dataset object, sample_ids, 
            feature_ids, checkpoint flag, experiment_description, etc. Check the definintion of the :py:meth::`run` method in this script to know about more arguments you can return from this method. 
            
            Additionally, you may add any entries to the dictionary that may aide in generating the pipeline, because these arguments are made available in the :py:meth::`generate_pipeline` method. 
            
            Args:
                **kwargs: The kwargs contain the command-line args, and the result from the previous workflows if the :py:meth::`run` method of this script is called from a pipeline of :py:class::`orthrus.core.pipeline.WorkflowManager` 
                        processes (check option 1 in the documentation of the :py:meth::`run` method). So these kwargs may be used to set the arguments for the return dictionary. 

            Returns: a dictionary of workflow arguments for the experiment
                    
                    It must contain :py:class:`orthrus.core.Dataset` object in 'ds' key
                    
                    It may optionally contain entries such as 'sample_ids', 'feature_ids', 'group_attr', 'supervised_attr' etc. Check the documentation of the :py:meth::`run` method of this script to know about 
                    more entries you can return. Also check the tutorials listed in the documentation of the :py:meth::`run` method for more ideas.

        b. :py:meth::`generate_pipeline`:
            This method generates and returns a :py:class::`orthrus.core.pipeline.Pipeline` object which is to be run. The location of the pipeline checkpoint is the artifacts directory of the 
                current mlflow run. The name of the pipeline is determined by the :py:attr::`workflow_name` arg of the :py:meth::`run` method of this script.

            Args:
                **kwargs: The kwargs contain the command-line args, arguments returned by  :py:meth::`get_workflow_args` method, and the result from the previous workflows if the :py:meth::`run` method of this 
                    script is called from a pipeline of :py:class::`orthrus.core.pipeline.WorkflowManager` processes (check option 1 in the documentation of the :py:meth::`run` method). These kwargs may be 
                    used to define the processes of the pipeline.
                
            Returns: a :py:class::`orthrus.core.pipeline.Pipeline` object

        Also, check out concrete examples on how to use this framework to run your orthrus pipeline in the tutorials 
        listed in the documentation of the :py:meth::`run` method of this script.
    '''

    # process command line args
    parser = utils.get_workflow_parser('run_pipeline_workflow', 'Runs a pipeline defined from a python script whose path is specified using experiment_module_path cmd-line arg.')
    args = vars(utils.process_args(parser))
    
    # setup logger, mlflow and ray runs
    logger = utils.setup_workflow_execution(**args, log_filename = 'execution.log')

    # load module and get workflow args
    experiment_module_path=args.pop('experiment_module_path')
    experiment_module = module_from_path(Path(experiment_module_path).stem, experiment_module_path)
    experiment_workflow_args = experiment_module.get_workflow_args(**args)
    utils.add_missing_kv_pairs(experiment_workflow_args, args)

    # run the workflow
    run(experiment_module_path, **args)

if __name__=="__main__":
    main()