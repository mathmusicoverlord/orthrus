"""Script for tuning an orthrus pipeline."""
# imports
from copy import deepcopy
import re
import json
import logging
import mlflow
import os
from pathlib import Path
import yaml
from urllib.parse import urlparse

import ray
import ray.air as air
from ray.tune.integration.mlflow import mlflow_mixin
import ray.tune as tune

import orthrus.core.helper as helper
from orthrus.core.pipeline import Pipeline
from orthrus.core.helper import module_from_path, extract_reconstruction_details_from_logger, reconstruct_logger_from_details
import orthrus.mlflow.modules.utils as utils


logger = logging.getLogger(__name__)

# TODO: 
# 1. Ray tune workflow does not support checkpointing


def generate_placement_group_for_nested_parallelization(max_cuncurrent_trials, 
                                                        num_cpus_for_job=-1, 
                                                        num_gpus_for_job=0, 
                                                        num_cpus_for_ray_trainable=1, 
                                                        num_gpus_for_ray_trainable=0, 
                                                        num_cpus_per_task=1, 
                                                        num_gpus_per_task=0):
        # this function considers the resource requirements, divides the resources by returning a placement group. (computes_resources_for_one_ray_tune_trial)

        # add resources for trainable as the first element of the list
        placement_group_list = [{"CPU": num_cpus_for_ray_trainable, "GPU": num_gpus_for_ray_trainable}]


        num_cpus_for_job, num_gpus_for_job = helper.process_resource_requirements_for_job_(num_cpus_for_job, num_gpus_for_job, num_gpus_per_task)


        # calculate remaining num cpus and gpus
        available_cpus = num_cpus_for_job - max_cuncurrent_trials * num_cpus_for_ray_trainable - 5
        available_gpus = num_gpus_for_job - max_cuncurrent_trials * num_gpus_for_ray_trainable

        try:
            assert available_cpus > 0
        except AssertionError as err:
            msg = f"CPUs are not available to run the ray tune trials. The method received {num_cpus_for_job} from process_resource_requirements_for_job_. "\
                    "After reserving {max_cuncurrent_trials * num_cpus_for_ray_trainable} CPUs for ray traiables, {available_cpus} were remaining."
            logger.error(msg)
            raise AssertionError(msg)

        if not(num_gpus_for_job == 0 and num_gpus_per_task == 0):
            try:
                assert available_gpus > 0
            except AssertionError as err:
                msg = f"GPUs are not available to run the ray tune trials. The method received {num_cpus_for_job} from process_resource_requirements_for_job_. "\
                    "After reserving {max_cuncurrent_trials * num_gpus_for_ray_trainable} GPUs for ray traiables, {available_gpus} were remaining."
                logger.error(msg)
                raise AssertionError(msg)


        # distribute available cpus and cpus among max_cuncurrent_trials. This is the amount of resources that are available to each trial
        available_cpus_for_nested_ray_job = available_cpus / max_cuncurrent_trials
        available_gpus_for_nested_ray_job = available_gpus / max_cuncurrent_trials


        # get the max number of ray tasks that can be run for each trial
        max_processes = helper.get_max_process_limit_(available_cpus_for_nested_ray_job, available_gpus_for_nested_ray_job, num_cpus_per_task, num_gpus_per_task)

        try:
            assert max_processes > 1
        except AssertionError as err:
            msg = f"Resources not available to run the ray tune trials. Out of {available_cpus_for_nested_ray_job} available CPUs and {available_gpus_for_nested_ray_job} avilable GPUs, "\
                f"{num_cpus_per_task} cpus and {num_gpus_per_task} gpus were requested for {max_cuncurrent_trials} ray tune trials."
            logger.error(msg)
            raise AssertionError(msg)

        
        # make sure that the resource allocation was done correctly, by ensuring that we didn't allocate more resources than are available.
        try:
            assert max_cuncurrent_trials*(max_processes * num_cpus_per_task + num_cpus_for_ray_trainable) <= num_cpus_for_job
        except AssertionError as err:
            msg = 'overallocated CPUs'
            logger.error(msg)
            raise AssertionError(msg)
        
        try:
            assert max_cuncurrent_trials*(max_processes * num_gpus_per_task + num_gpus_for_ray_trainable) <= num_cpus_for_job
        except AssertionError as err:
            msg = 'overallocated CPUs'
            logger.error(msg)
            raise AssertionError(msg)


        placement_group_list.extend([{"CPU": num_cpus_per_task, "GPU": num_gpus_per_task}] * max_processes)

        return [available_cpus_for_nested_ray_job, available_gpus_for_nested_ray_job], tune.PlacementGroupFactory(placement_group_list)



# def generate_placement_group_for_nested_parallelization(max_cuncurrent_trials, 
#                                                         num_cpus_for_job=-1, 
#                                                         num_gpus_for_job=0, 
#                                                         num_cpus_for_ray_trainable=1, 
#                                                         num_gpus_for_ray_trainable=0, 
#                                                         num_cpus_per_task=1, 
#                                                         num_gpus_per_task=0):
#         # this function considers the resource requirements, divides the resources by returning a placement group. (computes_resources_for_one_ray_tune_trial)

#         num_cpus_for_job, num_gpus_for_job = helper.process_resource_requirements_for_job_(num_cpus_for_job, num_gpus_for_job, num_gpus_per_task)

#         # reserve resources for main trainable processes
#         total_cpus_for_main_trainable_processes = max_cuncurrent_trials * num_cpus_for_ray_trainable
#         total_gpus_for_main_trainable_processes = max_cuncurrent_trials * num_gpus_for_ray_trainable
        
#         remaining_cpus = num_cpus_for_job - total_cpus_for_main_trainable_processes
#         remaining_gpus = num_gpus_for_job - total_gpus_for_main_trainable_processes

#         # 
#         cpus_available_to_tasks_of_one_ray_trainable = remaining_cpus / max_cuncurrent_trials  
#         gpus_available_to_tasks_of_one_ray_trainable = remaining_gpus / max_cuncurrent_trials 
        
#         try:
#             assert cpus_available_to_tasks_of_one_ray_trainable >= num_cpus_per_task
#         except AssertionError as err:
#             msg = f"CPUs are not available to run the parallel tasks of ray tune trials. There are {num_cpus_for_job} total CPUs available"\
#                     f", and after reserving {total_cpus_for_main_trainable_processes} CPUs for main processes of {max_cuncurrent_trials} ray traiables, {remaining_cpus} remaining CPUs were distributed among" \
#                     f" the ray trainables. This means that while each parallel task in the ray trainable requires {num_cpus_per_task} CPUs, only {cpus_available_to_tasks_of_one_ray_trainable} are available. "\
#                     "Try reducing the max_cuncurrent_trials or num_cpus_per_task."
#             logger.error(msg)
#             raise AssertionError(msg)

#         if not(num_gpus_for_job == 0 and num_gpus_per_task == 0):
#             try:
#                 assert gpus_available_to_tasks_of_one_ray_trainable >= num_gpus_per_task
#             except AssertionError as err:
#                 msg = f"GPUs are not available to run the parallel tasks of ray tune trials. There are {num_gpus_for_job} total GPUs available"\
#                         f", and after reserving {total_gpus_for_main_trainable_processes} GPUs for main processes of {max_cuncurrent_trials} ray traiables, {remaining_cpus} remaining GPUs were distributed among" \
#                         f" the ray trainables. This means that while each parallel task in the ray trainable requires {num_gpus_per_task} GPUs, only {gpus_available_to_tasks_of_one_ray_trainable} are available. "\
#                         "Try reducing the max_cuncurrent_trials or num_gpus_per_task."
#                 logger.error(msg)
#                 raise AssertionError(msg)

    
#          # add resources for trainable as the first element of the list
#         placement_group_list = [{"CPU": cpus_available_to_tasks_of_one_ray_trainable, "GPU": gpus_available_to_tasks_of_one_ray_trainable}]

#         return [cpus_available_to_tasks_of_one_ray_trainable, gpus_available_to_tasks_of_one_ray_trainable], tune.PlacementGroupFactory(placement_group_list)

# user defined functions
@mlflow_mixin
def set_description(path):
    mlflow.set_tag('mlflow.note.content',
                f"Tuning of pipeline {path}")



def trainable(config):
        from ray.air import session 
        args = ray.get(config['args_ref'])
        args.update(config)

        name = session.get_trial_name()

        # need to reconstruct logger
        # ray tunable are new processes (they do not have access to root logger that was created in the main process)
        logger_info = args['logger_info']

        # we are now running in a new process, the root logger in this new process does not have the file handlers.
        # So, we need to add the ofile handlers to the root logger, so that all logs get logged to respective files
        # when they are propagated to the root logger
        rootlogger =  helper.reconstruct_logger_from_details(logger_info, None)

        logger = logging.getLogger(f'{name}_pid-{os.getpid()}')
        
        experiment_module = module_from_path(args['experiment_module_name'], args['experiment_module_path'])
        
        #get environment variables
        experiment_variables : dict = experiment_module.get_experiment_variables()

        # load and slice dataset
        ds = utils.slice_dataset(experiment_variables)

        # generate pipeline from config
        pipeline: Pipeline = experiment_module.generate_pipeline(**args)
        sess = session._get_session().__dict__['_status_reporter'].__dict__
        pipeline.checkpoint_path = os.path.join(sess['_logdir'], 'pipeline.pkl')

        # run the pipeline on the data
        logger.info(f'Starting execution for trainable with id: {session.get_trial_id()}')

        pipeline.run(ds, checkpoint=args.get('checkpoint_trainables', False))
        
        logger.info('='*50)
        scores = experiment_module.score(pipeline, **args.get('score_args', {}))
        logger.info(scores)
        
        # return score
        return scores



def run(experiment_module, **kwargs):
    # check if an earlier execution exists!
    artifacts_dir = urlparse(mlflow.get_artifact_uri()).path

    iter_num = kwargs.get("iter", None)
    if iter_num is None:
        run_dir = artifacts_dir
        log_key = f'workflow_method_args'
    else:
        run_dir = os.path.join(artifacts_dir, f'run_{iter_num}')
        os.makedirs(run_dir, exist_ok=True)
        log_key = f'workflow_method_args_run_{iter_num}'

    # temoprarily remove the results while parameters are logged
    try:
        p = kwargs.pop('results_from_previous_workflows')
    except KeyError as e:
        p = None    
    
    mlflow.log_param(log_key,  yaml.dump(kwargs, allow_unicode=True, default_flow_style=False).replace('\n- ', '\n\n- '))

    if p is not None:
        kwargs['results_from_previous_workflows'] = p

    # only for mlflow logging
    experiment_variables : dict = experiment_module.get_experiment_variables()
    #remove ds before logging the params
    experiment_variables.pop('ds')
    #log experiment variables
    mlflow.log_params(experiment_variables)

    

    # update config with mlflow
    tracking_uri = mlflow.get_tracking_uri()
    active_run = mlflow.active_run()
    if active_run:
        experiment_id = active_run.info.experiment_id
    else:
        experiment_id = 0   
    kwargs['mlflow'] = {'experiment_id': experiment_id,
                        'tracking_uri': tracking_uri,
                        'workflow_artifacts_dir_path': run_dir}


    # for some reason passing the experiment_module here causes a runtime exception with no message
    kwargs['experiment_module_path'] = experiment_module.__file__
    kwargs['experiment_module_name'] = experiment_module.__name__
    kwargs['logger_info'] = helper.extract_reconstruction_details_from_logger(logging.getLogger())
    
    kwargs_ref = ray.put(kwargs)
    
    # extract config 
    config = kwargs['tune_config'].copy()
    config['args_ref'] = kwargs_ref


    if kwargs.get('tune_search_alg', None) is None:
        search_alg = experiment_module.search_alg(kwargs)

    else:
        search_alg = deepcopy(kwargs['tune_search_alg'])

    reg_compile = re.compile("TuneTrainable_*")
    dirs = []
    for dirname in next(os.walk(run_dir))[1]:
        if reg_compile.match(dirname):
            dirs += [dirname] 

    # NOTE: As of (10/21/22) tuner.restore only restarts pending + errored trials. It will not run the trials that weren't started, so the 
    # total number of trials may not equal to num_samples.
    # Removing the restore functionality for now. To put the functionality back, remove the next line (dirs=[])
    dirs =[]
    if len(dirs) == 1:
        path = os.path.join(run_dir, dirs[0])
        logger.info(f'Restoring Ray Tune run from the following location: {path}')
        tuner = tune.Tuner.restore(path=path, resume_unfinished = kwargs.get('resume_unfinished', True), restart_errored=kwargs.get('restart_errored', True))
    
    elif len(dirs) == 0: 
        logger.info(f'Starting a new Ray Tune run')        

        scheduler = experiment_module.scheduler(kwargs)

        tuner = tune.Tuner(tune.with_resources(trainable, kwargs['placement_group']),
                        param_space=config,
                        tune_config=tune.TuneConfig( num_samples=kwargs['num_samples'],
                                                        scheduler=scheduler,
                                                        search_alg=search_alg), 
                        run_config=air.RunConfig(local_dir = run_dir,
                                                stop={"training_iteration": kwargs.get('stopping_iteration', 1)},
                                                # log_to_file=(os.path.join(run_dir, "my_stdout.log"), os.path.join(run_dir, "my_stderr.log")),
                                                failure_config=air.FailureConfig(max_failures=1))
                                )    


    else:
        err_msg = f'Multiple ray tune trainable directories found at location: {run_dir}. Expected only one directory to be present.'
        logger.error(err_msg)
        raise Exception(err_msg)
    
    results = tuner.fit()
    best_results = results.get_best_result(mode=search_alg.mode, metric=search_alg.metric)    
    best_config = best_results.config

    if search_alg.mode == 'min':
        ascending = True
    else:
        ascending = False
    

    sorted_results = results.get_dataframe().sort_values(by=search_alg.metric, ascending= ascending)

    # tune_dir = tuner._local_tuner._experiment_checkpoint_dir
    sorted_results.to_csv(os.path.join(run_dir, 'results_df.csv'))


    # remove keys
    del kwargs['experiment_module_name']
    del kwargs['experiment_module_path']
    del kwargs['mlflow']
    del kwargs['logger_info']
    del best_config['args_ref']
    # log best hyperparameters
    with open(os.path.join(run_dir, 'best_config.json'), "w") as config_file:
        json.dump(best_config, config_file)


    args = {'results_location': run_dir,
            'tune_results': sorted_results,
            'tuner': tuner}
    
    try:
        experiment_module.process_results(**args)
    except AttributeError as e:
        logger.error(e, exc_info=True)
        # logger.error('The experiment module does not contain "process_results" method.')
    except:
        logger.error(e, exc_info=True)

    # set description
    set_description(experiment_module.__file__)

    try:
        mlflow.log_param(f'best_config_for_run_{iter_num}', best_config)
    except Exception as e:
        logger.error(f'Exception occured while trying to log the following parameters to MLFlow, {best_config}')
        logger.error(e, exc_info=True)

    best_config['all_results'] = sorted_results

    return None, best_config



if __name__ == "__main__":

    parser = utils.get_workflow_parser()
    args = utils.process_args(parser)
    
    experiment_module = module_from_path(Path(args.experiment_module_path).stem, args.experiment_module_path)
    
    args = vars(args)
    logger = utils.setup_workflow_execution(args, log_filename = 'execution.log')
    args.pop('experiment_module_path')

    experiment_workflow_args = experiment_module.get_tuning_workflow_args(args)
    
    run(experiment_module, **experiment_workflow_args)



# How to leverage fault tolerance in trial runner

# How does checkpointing work in Ray Tune: https://docs.ray.io/en/master/tune/tutorials/tune-checkpoints.html





# class OutputLogger:
#     def __init__(self, name="root", level="INFO"):
#         self.logger = logging.getLogger(name)
#         self.name = self.logger.name
#         self.level = getattr(logging, level)

#     def write(self, msg):
#         if msg and not msg.isspace():
#             self.logger.log(self.level, msg)

#     def flush(self): pass

# from typing import Dict, List
# from ray.tune.logger import LoggerCallback
# import json
# import os
# class CustomLoggerCallback(LoggerCallback):
#     """Custom logger interface"""

#     def __init__(self, filename: str = "log.txt"):
#         self._trial_files = {}
#         self._filename = filename

#     def log_trial_start(self, trial: "Trial"):
#         trial_logfile = os.path.join(trial.logdir, self._filename)
#         self._trial_files[trial] = open(trial_logfile, "at")

#     def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
#         if trial in self._trial_files:
#             self._trial_files[trial].write(json.dumps(result))

#     def on_trial_complete(self, iteration: int, trials: List["Trial"],
#                           trial: "Trial", **info):
#         if trial in self._trial_files:
#             self._trial_files[trial].close()
#             del self._trial_files[trial]




# class TuneTrainable(tune.Trainable):
#     def setup(self, config, args=None):
#         # config (dict): A dict of hyperparameters
#         self.config = config
#         if args is None:
#             self.args = {}
#         else:
#             self.args = args
#         self.args.update(self.config)
#         self.name = 'TuneTrainable' + '_' + self._experiment_id + '_' + self._trial_info._trial_id

#         # need to reconstruct logger
#         # ray tunable are new processes (they do not have access to root logger that was created in the main process)
#         self.logger_info = args['logger_info']

#         # we are now running in a new process, the base logger in this new process does not have the file handlers.
#         # So, make changes to the base logger
#         baselogger =  helper.reconstruct_logger_from_details(self.logger_info, None)

#     def step(self):  # This is called iteratively.

        
#         logger = logging.getLogger( f'{self.name}_pid-{os.getpid()}')
        
#         experiment_module = module_from_path(self.config['pipeline_name'], self.config['pipeline_path'])
        
#         #get environment variables
#         experiment_variables : dict = experiment_module.get_experiment_variables()

#         # load and slice dataset
#         ds = slice_dataset(experiment_variables)

#         # generate pipeline from config
#         pipeline: Pipeline = experiment_module.generate_pipeline(**self.args)
#         pipeline.checkpoint_path = os.path.join(self._logdir, 'pipeline.pkl')

#         # run the pipeline on the data
#         logger.info(f'Starting execution for trainable with id: {self._experiment_id}')

#         pipeline.run(ds, checkpoint=self.args.get('checkpoint_trainables', False))
        
#         logger.info('='*50)
#         scores = experiment_module.score(pipeline, **self.args.get('score_args', {}))
#         logger.info(scores)
#         # return score
#         return scores

    
#     # This class is not designed for checkpointing, the methods implemented here only exist to avoid runtime NotImplemented Exception
#     def save_checkpoint(self, tmp_checkpoint_dir):
#         return tmp_checkpoint_dir
    
#     def load_checkpoint(self, tmp_checkpoint_dir):
#         logger = logging.getLogger( f'{self.name}_pid-{os.getpid()}')
#         logger.info('called load_checkpoint')




# def generate_placement_group_for_single_level_parallelization(num_cpus_for_job, num_gpus_for_job, num_cpus_for_ray_trainable, num_gpus_for_ray_trainable, num_cpus_per_task, num_gpus_per_task):
#         # this function considers the resource requirements, divides the resources by returning a placement group. (computes_resources_for_one_ray_tune_trial)
        
#         # add resources for trainable as the first element of the list
#         placement_group_list = [{"CPU": num_cpus_for_ray_trainable, "GPU": num_gpus_for_ray_trainable}]


#         num_cpus_for_job, num_gpus_for_job = helper.process_resource_requirements_for_job_(num_cpus_for_job, num_gpus_for_job, num_gpus_per_task)


#         # # calculate remaining num cpus and gpus
#         available_cpus = num_cpus_for_job - num_cpus_for_ray_trainable - 5
#         available_gpus = num_gpus_for_job - num_gpus_for_ray_trainable


#         # get the max number of ray tasks that can be run for each trial
#         max_processes = helper.get_max_process_limit_(available_cpus, available_gpus, num_cpus_per_task, num_gpus_per_task)

#         try:
#             assert max_processes > 1
#         except AssertionError as err:
#             msg = f"Resources not available to run the ray tune trials. Out of {available_cpus} available CPUs and {available_gpus} avilable GPUs, "\
#                 f"{num_cpus_per_task} cpus and {num_gpus_per_task} gpus were requested for each of the {max_processes} ray tune trials."
#             logger.error(msg)
#             raise AssertionError(msg)

        
#         # make sure that the resource allocation was done correctly, by ensuring that we didn't allocate more resources than are available.
#         try:
#             assert (max_processes * num_cpus_per_task + num_cpus_for_ray_trainable) <= num_cpus_for_job
#         except AssertionError as err:
#             msg = 'overallocated CPUs'
#             logger.error(msg)
#             raise AssertionError(msg)
        
#         try:
#             assert (max_processes * num_gpus_per_task + num_gpus_for_ray_trainable) <= num_cpus_for_job
#         except AssertionError as err:
#             msg = 'overallocated CPUs'
#             logger.error(msg)
#             raise AssertionError(msg)

#         placement_group_list.extend([{"CPU": num_cpus_per_task, "GPU": num_gpus_per_task}] * max_processes)

#         return [available_cpus, available_gpus], tune.PlacementGroupFactory(placement_group_list)