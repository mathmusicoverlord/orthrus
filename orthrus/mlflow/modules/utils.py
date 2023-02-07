"""Module for utility functions"""

# imports
import ast
import os
import ray
import logging
import mlflow
import numpy as np
from orthrus.core.pipeline import Report, Score
from orthrus.core.helper import save_object
import pandas as pd
from pivottablejs import pivot_ui
import plotly.graph_objects as go
from urllib.parse import urlparse
import argparse
import yaml
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

def log_mlflow_params_and_add_softlinks(pipeline):
        # log mlflow run ids and experiment module path of each workflow in the main experiment
        # also create soflinks of workflow experiment directories in the main or default experiment's artifacts directory
        params = {}
        process_list = []
        
        artifact_dir_for_default_process = urlparse(mlflow.active_run()._info.artifact_uri).path
        for process in pipeline.processes:
                params[f'Run id - {process.process_name}'] = process.mlflow_run_id
                
                process_list.append(f'{process.process_name}')

                #add softlinks of the  WorkflowManagerProcess(s) to the 'default' process
                try:    
                        experiment_dir_path_for_process = urlparse(mlflow.get_experiment_by_name(process.process_name).artifact_location).path
                        run_dir_path_for_process = os.path.join(experiment_dir_path_for_process, process.mlflow_run_id)
                        
                        dst  = os.path.join(artifact_dir_for_default_process, process.process_name)
                        
                        if not os.path.exists(os.path.join(artifact_dir_for_default_process, process.process_name)):
                                os.symlink(src = run_dir_path_for_process, dst = dst, target_is_directory=True)
                        
                        # os.link(src = run_dir_path_for_process, dst = os.path.join(artifact_dir_for_default_process, process.process_name))
                except AttributeError as e:
                        logger.error(e, exc_info=True)

        # log mlflow parameters
        mlflow.log_param('Sequence of workflows', '\n------>\n'.join(process_list))
        mlflow.log_params(params)


def get_results_dir_and_file_name(workflow_name, iter_num):
    '''
    this method returns results directory path, name of pipeline pkl file and a key for logging 
    parameters in mlflow
    '''

    filename = workflow_name
    artifacts_dir = urlparse(mlflow.get_artifact_uri()).path
    if iter_num is None:
        run_dir = artifacts_dir
        log_key = f'workflow_method_args'
    else:
        run_dir = os.path.join(artifacts_dir, f'run_{iter_num}')
        os.makedirs(run_dir, exist_ok=True)
        filename = f'{filename}_{iter_num}_.pickle'
        log_key = f'workflow_method_args_run_{iter_num}'
    
    return run_dir, filename, log_key

def log_kwargs(kwargs):
    ds = kwargs.pop('ds') if 'ds' in kwargs else None
    
    try:
        for k, v in kwargs.items():
            mlflow.log_param(k,  v)
    except MlflowException as e:
        param_dir = os.path.join(urlparse(mlflow.get_artifact_uri()).path, '..', 'params')
        param_file = os.path.join(param_dir, f'{k}')
        # read lines from param file
        with open(param_file, 'r') as f:
            old_value = f.readlines()[0]
        
        os.remove(param_file)
        new_value = old_value + ' | ' + v
        logger.info(f'Found existing parameter :{k} with old value: {old_value}. Updating with new value: {new_value}')
        mlflow.log_param(k,  new_value)

    if ds is not None:
        kwargs['ds'] = ds 


def reformat_kwargs(kwargs):
    '''
    if :py:attr:`kwargs` contains a 'results_from_previous_workflows' key, this method\\
        1. pops the dictionary in 'results_from_previous_workflows' key\\
        2. logs the :py:attr:`kwargs` to mlflow with key=:py:attr:`log_key`\\
        3. updates :py:attr:`kwargs` with the dictionary that was originally in 'results_from_previous_workflows' key\\
        i.e. all the k-v pairs that were present in 'results_from_previous_workflows' dict are now\\
        available directly in :py:attr:`kwargs`.
    '''
    
    old_results = kwargs.pop('results_from_previous_workflows') if 'results_from_previous_workflows' in kwargs else None
    kwargs_copy = kwargs.copy()
    if old_results is not None:
        kwargs_copy.update(old_results)
    
    return kwargs_copy




    
def add_missing_kv_pairs(source, target):
    '''
    Adds the unique key-value pairs from source dict to target dict
    '''
    remaining_kwargs_keys = [x for x in list(source.keys()) if x not in target.keys()]
    for k in list(remaining_kwargs_keys):
            target[k] = source[k]


def check_type(x):
    # throw exception if x is not integer or none type
    if not (isinstance(x, int) or x is None or x == 'None'):
        raise TypeError(f'{x} is not an integer or None type')


def get_workflow_parser(parser_name, parser_description, add_exp_module_path=False):
    '''
    Returns a argmument parser object. Ths parser 'knows' about the most critical arguments which are required for 
    all experiments. The known arguments are:
    1. experiment_module_path

    2. run_id 

    3. experiment_id 

    4. num_cpus_for_job 

    5. num_gpus_for_job 

    6. local_mode
    '''

    # command line arguments
    parser = argparse.ArgumentParser(parser_name, description=parser_description)
    if add_exp_module_path:
        parser.add_argument("--experiment_module_path",
                        type=str, help="File path to the python module. Please check the documentation of the main method of the script to see the requirements for the python module to be valid.")

    parser.add_argument("--run_id",  default=None,
                        type=str, help="In case of a re-run, this argument provides the mlflow run id of the experiment to be re-run.")

    parser.add_argument("--experiment_id", default="0",
                        type=str, help="In case of a re-run, this argument provides the mlflow experiment id of the experiment to be re-run.")

    parser.add_argument("--num_cpus_for_job", type=check_type, default=None, help="How many cpus to use for the whole job")

    parser.add_argument("--num_gpus_for_job", type=check_type, default=0, help="How many gpus to use for the whole job")

    parser.add_argument("--local_mode", default='False', help="Whether to run ray in local mode. Only 'True' is considered True, all other values will be considered False.")

    return parser


def process_args(parser):
    '''
    This method processes the known and unknown command line arguments in the parser object. Amongst the unknown arguments,
    any argument with the 'key=value' format is added to the arguments 'known' to the parser.

    Returns:
        All command line arguments that are either known to the parser or matches the 'key=value' format
    '''


    args, unknown = parser.parse_known_args()

    remaining_args = {}
    unknown = (x for x in (a.split('=') for a in unknown) if len(x) >1)
    for k, v in ((k.lstrip('-'), v) for k,v in unknown):
        try:
            remaining_args[k] = ast.literal_eval(v)
        except ValueError:
            remaining_args[k] = v

    args_dict = vars(args)
    args_dict.update(remaining_args)

    return args

def setup_workflow_execution(experiment_id:str="0",
                            run_id:str = None,
                            local_mode:bool=False,
                            num_cpus_for_job:int=None,
                            num_gpus_for_job:int=None,
                            log_filename:str = 'global.log',
                            **kwargs):
    '''
    This method is responsible for 
        1. Starting the correctly mlflow run
        2. Setting up and formatting the root logger. A file handler is added to the root logger, 
           the location of the log file is the artifacts directory of the mlflow run, and it's name is
           determined by :py:attr::`log_filename`
        3. starting ray tune

    Args:
        :py:attr::`experiment_id` (str): mlflow expeirment_id of the experiment. defaults to 0.
        :py:attr::`run_ud` (str): mlflow run_id of the experiment. Defaults to None, which means start a new run. 
        :py:attr::`local_mode` (bool): whether to start ray in local_mode
        :py:attr::`num_cpus_for_job`: How many cpus to use initialize ray with. Defaults to None, which means use all available CPUs.
        :py:attr::`num_gpus_for_job`: How many gpus to use initialize ray with. Defaults to None, which means use all available GPUs
        :py:attr::`log_filename`: Name of the log file.

    Returns:
        Instance of the root logger
    '''
        
    msg = ''
    # if this environment variable is set, it means that this script was run using mlflow run command.
    # In this case we simple need to start the mlflow run and it automatically knows which run to start
    if os.environ.get('MLFLOW_EXPERIMENT_ID', None) is not None:
            msgs = ['The script execution was started from "mlflow run" command, experiment and run id provided by mlflow cli.']
            active_run =  mlflow.start_run()
    else:
            msgs = ['Mlflow experiment_id and run_id information was passed by command-line argument to python script with values:',
                    f'experiment_id: {experiment_id}',
                    f'run_id: {run_id}']
            mlflow.set_experiment(experiment_id=experiment_id)
            active_run =  mlflow.start_run(run_id = run_id, experiment_id=experiment_id)


    format_string = '%(asctime)s - %(name)s - %(levelname)s : \t %(message)s'
    logging.basicConfig(level=logging.INFO, format=format_string, datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(urlparse(mlflow.get_artifact_uri()).path, log_filename), 'a')
    handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(handler)
    for msg in msgs: 
        logger.info(msg) 

    if run_id is None:
        run_id = active_run.info.run_id
        logger.info(f'Starting workflow execution, with new mlflow run_id: {run_id}')
    else:
        logger.info(f'Restarting workflow execution, with run_id {run_id}')

    
    local_mode=ast.literal_eval(str(local_mode))
    if local_mode:
        logger.info('"local_mode" commandline argument value True passed, starting ray in local mode')

    address = os.environ.get("ip_head", None)
    logger.info('Starting ray tune with the following parameters:')
    logger.info(f'local_mode: {local_mode}')
    
    if address is not None:
        logger.info(f'address: {address}')
        ray.init(address=address, local_mode = local_mode)
    else:
        logger.info(f'num_cpus_for_job: {num_cpus_for_job}')
        logger.info(f'num_gpus_for_job: {num_gpus_for_job}')
        ray.init(local_mode = local_mode, num_cpus = num_cpus_for_job, num_gpus = num_gpus_for_job)
    
    return logger

        



def condense_nested_dict(in_dict: dict, out_dict:dict, prev_key=None) -> None:
    """Extracts nested dictionary values into a condensed dictionary"""
    for k,v in in_dict.items():
        if prev_key is not None:
            new_key = '.'.join([prev_key, k])
        else:
            new_key = k
        if isinstance(v, dict):
            condense_nested_dict(v, out_dict=out_dict, prev_key=new_key)
        else:            
            out_dict.update({new_key : v})

def score_violin_plot(score_frame: pd.DataFrame) -> None:
    """
    Generates a plotly boxplot figure for a dataframe of scores of shape
    (n_runs, n_scores)
    """
    # initialize Plotly figure
    fig = go.Figure()
    
    # generate box plot for each feature
    for score in score_frame:
        fig.add_trace(go.Violin(y=score_frame[score].values, name=score, visible='legendonly', box_visible=True, meanline_visible=True))
    
    return fig

def log_report_scores(report: Report) -> None:
    """Logs classification metrics from an orthrus Report class instance."""

    # extract scores from report
    scores = report.report()

    # loop through train/test and train/valid/test type splits
    for split_type in scores.keys():
        split_scores: pd.DataFrame = scores[split_type]
        score_types: list = split_type.split('_')
        score_types = [score_type.capitalize() for score_type in score_types]
        for score_type in score_types:
            
            # compute bsr
            try:
                split_scores[f"{score_type}:BSR"] = \
                split_scores.filter(regex=f"{score_type}:.+:Recall",
                                    ).drop(columns=[f"{score_type}:macro avg:Recall",
                                                    f"{score_type}:weighted avg:Recall"]).mean(axis=1)
            except KeyError:
                pass

        # log individual scores
        for batch in split_scores.index:
            for metric in split_scores.columns:
                metric_split = metric.split(':')
                score = split_scores.at[batch, metric]
                if not pd.isna(score) and ~np.isnan(score):
                    mlflow.log_metric(f"{split_type}.{batch}.{'.'.join(metric_split)}", score)

        split_scores = split_scores.fillna(0)
        # log means, standard dev, min, max
        split_scores_stats = {} 
        split_scores_stats['Mean'] = split_scores.mean(axis=0)
        split_scores_stats['Std'] = split_scores.std(axis=0)
        split_scores_stats['Min'] = split_scores.min(axis=0)
        split_scores_stats['Max'] = split_scores.max(axis=0)
        for stat, stat_scores in split_scores_stats.items():
            for metric in stat_scores.index:
                metric_split = metric.split(':')
                score = stat_scores[metric]
                if not pd.isna(score) and  ~np.isnan(score):
                    mlflow.log_metric(f"{split_type}.{stat}.{'.'.join(metric_split)}", score)
        
        # store violin plot
        if split_scores.size > 0:
            fig: go.Figure = score_violin_plot(split_scores)
            violin_path = f"/tmp/{split_type}.violin.kartikay.html"
            fig.write_html(violin_path)
            mlflow.log_artifact(violin_path)
        
            # store html table of scores
            scores_path = f"/tmp/{split_type}.scores.kartikay.html"
            pivot_ui(split_scores, scores_path)
            mlflow.log_artifact(scores_path)
        

    return

def log_confusion_stats(confusion_mats: Score) -> None:
    """Logs classification metrics from an orthrus Score class instance containing a confusion matrix."""

    # extract scores from report
    split_scores = confusion_mats.condense_scores()

    for split in split_scores.keys():

        # extract dataframe of confusion mats
        cms = split_scores[split]

        if cms.size > 0:

            # initialize outer report
            outer_report = pd.DataFrame()

            # loop through split type
            for split_type in cms.columns:

                # initialize inner report
                inner_report = pd.DataFrame()
            
                # loop through the batches
                for batch in cms.index:
            
                    # grab the confusion matrix
                    cm = cms.loc[batch, split_type]

                    # get stats
                    stats = stats_from_conf_mat(cm.values)
                    stat_scores = np.array(list(stats.values())).reshape(1, -1)

                    # append to output report
                    report_columns = [f"{split_type}.{k}" for k in stats.keys()]
                    report_row = pd.DataFrame(index=[batch], columns=report_columns, data=stat_scores)
                    inner_report = pd.concat((inner_report, report_row), axis=0)

                    # log the stats
                    stats = {f"{split}.{split_type}.{batch}.{k}": v for k, v in stats.items()}
                    mlflow.log_metrics(stats)
                
                # append inner report
                outer_report = pd.concat((outer_report, inner_report), axis=1)

                # sum across batches
                cm = sum([cms.loc[batch, split_type] for batch in cms.index])

                # get stats
                stats = stats_from_conf_mat(cm.values)

                # log the stats
                stats = {f"{split}.{split_type}.Mean.{k}": v for k, v in stats.items()}
                mlflow.log_metrics(stats)


            # store violin plot
            fig: go.Figure = score_violin_plot(outer_report.dropna(axis=1))
            violin_path = f"/tmp/{split}.violin.kartikay.html"
            fig.write_html(violin_path)
            mlflow.log_artifact(violin_path)

    return

def stats_from_conf_mat(cm: np.ndarray):
    """Computes basic classification stats from a binary confusion matrix."""

    # extract tp, tn, fp, tn
    tn, fp, fn, tp = cm.ravel()

    # initialize output
    stats = {}

    # compute stats
    stats['TPR'] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    stats['TNR'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    stats['PPV'] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    stats['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    stats['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    stats['FNR'] = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    stats['FDR'] = fp / (fp + tp) if (fp + tp) > 0 else np.nan
    stats['FOR'] = fn / (fn + tn) if (fn + tn) > 0 else np.nan
    stats['ACC'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    stats['BSR'] = (stats['TPR'] + stats['TNR']) / 2
    stats['F1'] = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan

    return stats



import os
import json
import mlflow

# def slice_dataset(ds, sample_ids=None, feature_ids=None, **kwargs):
#     '''
#     This method slices the datasets using sample_ids and feature_ids
#     Args:
#         samples_ids: must be a valid pandas query or be comatible with sample_ids of orthrus.core.dataset.slice_dataset method

#         feature_ids: must be a valid pandas query or be comatible with feature_ids of orthrus.core.dataset.slice_dataset method 
#     '''
#     # load and slice dataset
#     if sample_ids is not None:
#         if type(sample_ids)==str:
#             sample_ids = ds.metadata.query(sample_ids).index

#     if feature_ids is not None:
#         if type(feature_ids)==str:
#             feature_ids = ds.vardata.query(feature_ids).index
    
#     ds = ds.slice_dataset(sample_ids = sample_ids, feature_ids = feature_ids)

#     return ds


def log_pipeline_processes(dir, pipeline):
    '''
    this methods logs the processes of an orthrus pipeline (and their attrs) as an 
    json artifact file
    '''
    all_processes_params = {}
    for process in pipeline.processes:
        current_process_params = {}
        try:
            obj = process.process.__dict__

        except AttributeError as e:
            obj  = process.__dict__
        
        for k, v in obj.items():
            current_process_params[k] = str(v)

        all_processes_params[process.process_name] = current_process_params

    filepath = os.path.join(dir, 'process_params.json')
    save_object(json.dumps(all_processes_params), filepath, overwrite=True)