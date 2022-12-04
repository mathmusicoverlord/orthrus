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

def set_description(description):
    mlflow.set_tag('mlflow.note.content', description)



def trainable(config):
        from ray.air import session 
        args = ray.get(config['args_ref'])
        del config['args_ref']
        # add the search space sample(config) to args
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
        
        # get and slice dataset
        ds = args['ds']
        ds = ds.slice_dataset(sample_ids = args.get('sample_ids', None),
                              feature_ids = args.get('feature_ids', None))

        experiment_module = module_from_path(args['experiment_module_name'], args['experiment_module_path'])
        
        # generate pipeline from args
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
        session.report(scores)



def run(experiment_module_path, 
        tune_config,
        num_samples,
        placement_group,
        tune_search_alg=None,
        tune_scheduler=None,
        resume_unfinished = True,
        restart_errored = True,
        max_failures=1,
        workflow_name: str=None, 
        n_iter=None, 
        experiment_description: str=None,
        workflow_process_results_handle=None,
        **kwargs):

    # generate module
    experiment_module = module_from_path(Path(experiment_module_path).stem, experiment_module_path)

    run_dir, _, log_key = utils.get_results_dir_and_file_name(workflow_name, experiment_module, n_iter)
    utils.reformat_kwargs_about_results_from_previous_process(kwargs, log_key)

    # for some reason passing the experiment_module here causes a runtime exception with no message
    # so the solution that works is to pass experiment module's details and recreate it in the trainable
    kwargs['experiment_module_path'] = experiment_module.__file__
    kwargs['experiment_module_name'] = experiment_module.__name__
    kwargs['logger_info'] = helper.extract_reconstruction_details_from_logger(logging.getLogger())
    
    # extract the search space definition (config)
    config = tune_config.copy()

    #put the kwargs on ray store, this will be accessed in the trainable using kwargs_ref
    kwargs_ref = ray.put(kwargs)
    
    # add ray reference to config
    config['args_ref'] = kwargs_ref

    # find if a tune trainable directory already exists, if it does it means that a previous run failed which can be restored
    reg_compile = re.compile("trainable_*")
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
        tuner = tune.Tuner.restore(path=path, resume_unfinished = resume_unfinished, restart_errored=restart_errored)
    
    elif len(dirs) == 0: 
        logger.info(f'Starting a new Ray Tune run')        

        tuner = tune.Tuner(tune.with_resources(trainable, placement_group),
                        param_space=config,
                        tune_config=tune.TuneConfig( num_samples=num_samples,
                                                        scheduler=tune_scheduler,
                                                        search_alg=tune_search_alg), 
                        run_config=air.RunConfig(local_dir = run_dir,
                                                # stop={"training_iteration": kwargs.get('stopping_iteration', 1)},
                                                # log_to_file=(os.path.join(run_dir, "my_stdout.log"), os.path.join(run_dir, "my_stderr.log")),
                                                failure_config=air.FailureConfig(max_failures=max_failures))
                                )    


    else:
        err_msg = f'Multiple ray tune trainable directories found at location: {run_dir}. Expected only one directory to be present.'
        logger.error(err_msg)
        raise Exception(err_msg)
    
    # run tuner
    results = tuner.fit()

    #get best results
    best_results = results.get_best_result(mode=tune_search_alg.mode, metric=tune_search_alg.metric)    
    best_config = best_results.config

    # sort results
    ascending = True if tune_search_alg.mode == 'min' else False
    sorted_results = results.get_dataframe().sort_values(by=tune_search_alg.metric, ascending= ascending)

    # save the sorted results
    sorted_results.to_csv(os.path.join(run_dir, 'results_df.csv'))


    # remove keys that we added for our convenience
    del kwargs['experiment_module_name']
    del kwargs['experiment_module_path']
    del kwargs['logger_info']

    # log best hyperparameters
    with open(os.path.join(run_dir, 'best_config.json'), "w") as config_file:
        json.dump(best_config, config_file)
    
    # process the results
    if workflow_process_results_handle is not None:
        workflow_process_results_handle(results_location = run_dir, results = sorted_results)

    # set description
    if experiment_description is not None:
        set_description(experiment_description)

    try:
        mlflow.log_param(f'best_config_for_run_{n_iter}', best_config)
    except Exception as e:
        logger.error(f'Exception occured while trying to log the following parameters to MLFlow, {best_config}')
        logger.error(e, exc_info=True)

    best_config['all_results'] = sorted_results

    return None, best_config



if __name__ == "__main__":

    # process command line args
    parser = utils.get_workflow_parser()
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