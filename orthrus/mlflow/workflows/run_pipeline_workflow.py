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

class OrthrusPipelineWorkflow():

    def __init__(self, 
                get_workflow_args_handle,
                generate_pipeline_handle,
                process_workflow_results_handle=None):
        '''

        This workflow class provides a general-purpose abstraction for creating, runing and processing the results of an orthrus pipeline. It also tracks experiment information to mlflow.

        For a standalone usage, find an in-depth tutorial at ****************, and for use with :py:class::`orthrus.core.pipeline.WorkflowManager`, find the tutorial at *************.

        Parameters:
            1) :py:attr::`get_workflow_args_handle`: Handle to any method that returns a dictionary of arguments to be used for running the workflow. The returned dictionary can contain infomation such as the dataset object, sample_ids, 
                feature_ids, checkpoint flag, experiment_description, etc. Additionally, you may add any entries to the dictionary that may aide in generating the pipeline, because these arguments 
                are made available in the :py:meth::`generate_pipeline_handle` method. Check the requirements for this method below:
                
                Args:
                    **kwargs: The kwargs may be used to set the arguments for the return dictionary. When running a pipeline of :py:class::`orthrus.core.pipeline.WorkflowManager`, these kwargs always contain the command-line args and the 
                    results from earlier workflows. To access the results of a particular ealier workflow, use the workflow_name key to retrive the results from kwargs. To learn how to send command-line arguments to this method when manually calling
                    the :py:meth::`OrthrusPipelineWorkflow.run` method, check the tutorials listed above.

                Returns: a dictionary of workflow arguments for the experiment
                        
                        It must contain the following key-value pairs:

                            a) 'ds' (orthrus.core.dataset.Dataset) : Dataset object.  

                            b) 'workflow_name' (str): Name of the pipeline checkpoint file.


                        Optional key-value pairs:

                            'sample_ids' (str): The experiment will be restricted to just these samples from the ds. The value must be compatible with :py:meth::`orthrus.core.dataset.DataSet.slice_dataset` method.
                                                defaults to using all samples from ds.
                            
                            'feature_ids': (str): The experiment will be restricted to just these features from the ds. The value must be compatible with :py:meth::`orthrus.core.dataset.DataSet.slice_dataset` method.
                                                defaults to using all features from ds.
                            
                            'checkpoint' (bool): Whether to save the pipeline checkpoint to disk or not. defaults to True.

                            'experiment_description (str): Description of the experiment for mlflow tracking. defaults to None, which means don't add the description to mlflow.
                            
                        
                        It may optionally contain entries such as 'group_attr', 'supervised_attr' or any attributes that you may want to receive in :py:meth::`generate_pipeline_handle` method. Check the tutorials linked above for concrete examples.


            2) :py:attr::`generate_pipeline_handle`: Handle to a method that generates and returns a :py:class::`orthrus.core.pipeline.Pipeline` object which is to be run. The location of the pipeline checkpoint is the artifacts directory of the 
                    current mlflow run. The name of the pipeline is determined by the `workflow_name` entry from the dictionary returned by :py:meth::`get_workflow_args_handle`. Check the requirements for this method below:
                
                Args:
                    **kwargs: The kwargs may be used to define the processes of the pipeline. The kwargs always contain all the entries from the dictionary returned by :py:meth::`get_workflow_args_handle` method. Additionallty, when running a pipeline of 
                    :py:class::`orthrus.core.pipeline.WorkflowManager`, kwargs also always contain the command-line args, and the results from earlier workflows. To access the results of a particular ealier workflow, use the workflow_name key to retrive 
                    the results from kwargs. To learn how to send command-line arguments to this method when manually calling the :py:meth::`OrthrusPipelineWorkflow.run` method, check the tutorials listed above.
                    
                Returns: a :py:class::`orthrus.core.pipeline.Pipeline` object

            3) :py:attr::`process_workflow_results_handle`: Handle to a method that processes the results of the pipeline after it has finished its execution. For instance, this method could save csv files and/or generate plots 
                and html files etc. Check the requirements for this method below:
                
                Args:
                    :py:attr::`results_location`: path of working directory where the results can be stored. defaults to the artifacts directory of the current mlflow run.
                    :py:attr::`pipeline`: a :py:class::`orthrus.core.pipeline.Pipeline` object which has finished its execution.
    '''
    
        self.generate_pipeline = generate_pipeline_handle
        self.get_workflow_args = get_workflow_args_handle
        self.process_workflow_results = process_workflow_results_handle
        self.initialize_args = True

    def run(self,
            **kwargs):

        '''
        This methods gets the experiment args, generates the pipeline and processes the results.

        Args:
            **kwargs: The kwargs are be used to send command-line arguments, and the results of earlier workflows when called from a pipeline of :py:class::`orthrus.core.pipeline.WorkflowManager`.

        Returns:
            tuple of (finished :py:class::`orthrus.core.pipeline.Pipeline`, its results dictionary)
        '''  

        reformatted_kwargs = utils.reformat_kwargs(kwargs) 
        if self.initialize_args:
            self.args = self.get_workflow_args(**reformatted_kwargs)
        else:
            self.args = kwargs

        for key in ['ds', 'workflow_name']:
            all_mandatory_keys_present = True
            if key not in self.args:
                all_mandatory_keys_present = False
                logger.error(f'The argument dictionary returned by the get_experiment_args_handle method does not contain the key: {key}')

            if all_mandatory_keys_present is False:
                import sys
                sys.exit(0)    

        
        ds = self.args.pop('ds')
        experiment_description = self.args.get('experiment_description', None)
        workflow_name = self.args['workflow_name']
        n_iter = self.args.get('n_iter', None)
        sample_ids = self.args.get('sample_ids', None)
        feature_ids = self.args.get('feature_ids', None)
        checkpoint = self.args.get('checkpoint', True)

        run_dir, filename, log_key = utils.get_results_dir_and_file_name(workflow_name, n_iter)
        
        # create checkpoint file path
        checkpoint_path = os.path.join(run_dir, filename)
        self.args['checkpoint_path']=checkpoint_path  

        utils.log_kwargs(self.args)
        utils.add_missing_kv_pairs(reformatted_kwargs, self.args)

        ds = ds.slice_dataset(sample_ids = sample_ids,
                                feature_ids = feature_ids)

        #log experiment variables
        # mlflow.log_params(experiment_variables)

        # get pipeline
        pipeline : Pipeline = self.generate_pipeline(**self.args)

        #log parameters of the processes of the pipeline as artifacts
        utils.log_pipeline_processes(run_dir, pipeline)

        # run the pipeline on the data
        pipeline.run(ds, checkpoint=checkpoint)

        # process the results of the pipeline
        if self.process_workflow_results is not None:        
            self.process_workflow_results(results_location = run_dir, results = pipeline)

        # set description 
        self._set_description(experiment_description)

        return pipeline, pipeline.results_
            

    def _set_description(self, description):
        if description is not None:
            mlflow.set_tag('mlflow.note.content', description)


  


if __name__ == '__main__':

        '''
        This script provides a general-purpose abstraction for creating, runing and processing the results of an orthrus pipeline. It requires a mandatory command-line argument - :py:attr::`experiment_module_path` - which specifies the 
        location of a python module. Please run "python /path/to/file/run_pipeline_workflow.py -h" to know about other command-line arguments you can use. The python module located at :py:attr::`experiment_module_path` must contain methods to get the 
        arguments of the experiment workflow, to generate the pipeline to be run, and optionally a method to process the results. The exact specifications for these methods are in the documentation of the :py:class::`OrthrusPipelineWorkflow` class. 
        Also, check out concrete examples on how to use this framework to run your orthrus pipeline in the tutorials there.
        '''        

        # process command line args
        parser = utils.get_workflow_parser('run_pipeline_workflow', 'Runs a pre-defined defined orthrus pipeline', add_exp_module_path=True)
        args = vars(utils.process_args(parser))
        
        # setup logger, mlflow and ray runs
        utils.setup_workflow_execution(**args, log_filename = 'execution.log')
        
        experiment_module_path=args.pop('experiment_module_path')
        experiment_module = module_from_path(Path(experiment_module_path).stem, experiment_module_path)

        try:
            process_workflow_results_handle = experiment_module.process_results
        except AttributeError as e:
            logger.error(f'The experiment module located at {experiment_module_path} does not contain a process_results method.')
            logger.error('Results for the workflow cannot be processed')

        workflow = OrthrusPipelineWorkflow(
                        get_workflow_args_handle = experiment_module.get_workflow_args,
                        generate_pipeline_handle = experiment_module.generate_pipeline,
                        process_workflow_results_handle = process_workflow_results_handle,
                        )

        workflow.run(**args)
        pass