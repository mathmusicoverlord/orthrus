"""Module for utility functions"""

# imports
import logging
import mlflow
import numpy as np
from orthrus.core.pipeline import Report, Score
import pandas as pd
from pivottablejs import pivot_ui
import plotly.graph_objects as go
from urllib.parse import urlparse
import argparse


def get_workflow_parser():
    # command line arguments
    parser = argparse.ArgumentParser("Run Pipeline", description="Runs a pre-defined pipeline.")
    parser.add_argument("--experiment_module_path",
                        type=str, help="File path to the python file which defines a pipeline of WorkflowManager processes")

    parser.add_argument("--run_id",  default=None,
                        type=str, help="In case of a re-run, this argument provides the mlflow run id of the experiment to be re-run.")

    parser.add_argument("--experiment_id", default="0",
                        type=str, help="In case of a re-run, this argument provides the mlflow experiment id of the experiment to be re-run.")

    parser.add_argument("--num_cpus_for_job", type=int, default=None, help="How many cpus to use for the whole job")

    parser.add_argument("--num_gpus_for_job", type=int, default=None, help="How many gpus to use for the whole job")

    # Workaround: type of local_mode is string instead of boolean because the run command in the mlflow project file will always add a -P local_mode={local_mode}
    # argument, and that (presence of any value) is interpreted as True by ArgumentParse. Setting local_mode to False is done by not providing  any value in the 
    # command line argument, i.e. an absense of this flag means False. 
    parser.add_argument("--local_mode", default='False', help="Whether to run ray in local mode. Any value other than 'False' is considered True.")

    return parser


def process_args(parser):
    args, unknown = parser.parse_known_args()

    remaining_args = {}
    unknown = (x for x in (a.split('=') for a in unknown) if len(x) >1)
    for k, v in ((k.lstrip('-'), v) for k,v in unknown):
        remaining_args[k] = v

    args_dict = vars(args)
    args_dict.update(remaining_args)

    return args

def setup_workflow_execution(args, log_filename = 'global.log'):
        import os
        import ray
        
        # if this environment variable is set, it means that this script was run using mlflow run command.
        # In this case we simple need to start the mlflow run and it automatically knows which run to start
        experiment_id = os.environ.get('MLFLOW_EXPERIMENT_ID', None)

        msg = ''
        if experiment_id is not None:
                msgs = ['The script execution was started from "mlflow run" command, experiment and run id provided by mlflow cli.']
                active_run =  mlflow.start_run()
        else:
                msgs = ['Mlflow experiment_id and run_id information was passed by command-line argument to python script with values:',
                        f'experiment_id: {args["experiment_id"]}',
                        f'run_id: {args["run_id"]}']
                mlflow.set_experiment(experiment_id=args['experiment_id'])
                active_run =  mlflow.start_run(run_id = args['run_id'], experiment_id=args['experiment_id'])

        run_id = active_run.info.run_id

        format_string = '%(asctime)s - %(name)s - %(levelname)s : \t %(message)s'
        logging.basicConfig(level=logging.INFO, format=format_string, datefmt='%m/%d/%Y %H:%M:%S')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(urlparse(mlflow.get_artifact_uri()).path, log_filename), 'a')
        handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(handler)
        for msg in msgs: 
            logger.info(msg) 
        logger.info(f'Starting workflow execution for module located at path {args["experiment_module_path"]}')
       
        local_mode=False
        if args['local_mode'] != 'False':
            logger.info('"local_mode" commandline argument value True passed, starting ray in local mode')
            local_mode=True

        address = os.environ.get("ip_head", None)
        # num_cpus = args.get('num_cpus', None)
        # if num_cpus is not None: 
        #     num_cpus = int(num_cpus) 

        # num_gpus = args.get('num_gpus', None)
        # if num_gpus is not None: 
        #     num_gpus = int(num_gpus) 
        ray.init(address=address, local_mode = local_mode, num_cpus = args.get('num_cpus_for_job', None), num_gpus = args.get('num_gpus_for_job', None))
        
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
                if ~np.isnan(score):
                    mlflow.log_metric(f"{split_type}.{batch}.{'.'.join(metric_split)}", score)

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
                if ~np.isnan(score):
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

def slice_dataset(experiment_variables):
    # load and slice dataset
    ds = experiment_variables['ds']
    if experiment_variables['sample_id_query'] is not None:
        if type(experiment_variables['sample_id_query'])==str:
            sample_ids = ds.metadata.query(experiment_variables['sample_id_query']).index
        else:
            sample_ids = experiment_variables['sample_id_query']
    else:
        sample_ids = None

    if experiment_variables['feature_id_query'] is not None:
        if type(experiment_variables['feature_id_query'])==str:
            feature_ids = ds.vardata.query(experiment_variables['feature_id_query']).index
        else:
            feature_ids = experiment_variables['feature_id_query']
    else:
        feature_ids = None
    
    ds = ds.slice_dataset(sample_ids = sample_ids, feature_ids = feature_ids)

    return ds


def log_pipeline_processes(pipeline):
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

    file_path = '/tmp/process_params.json'
    with open(file_path, 'w') as f:
        json.dump(all_processes_params, f)

    mlflow.log_artifact(file_path)
    os.remove(file_path)