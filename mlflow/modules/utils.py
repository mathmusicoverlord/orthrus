"""Module for utility functions"""

# imports
from orthrus.core.pipeline import Report, Score
from pandas import DataFrame
from numpy import ndarray
import mlflow
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pivottablejs import pivot_ui

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

def score_violin_plot(score_frame: DataFrame) -> None:
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
            violin_path = f"tmp/{split_type}.violin.html"
            fig.write_html(violin_path)
            mlflow.log_artifact(violin_path)
        
            # store html table of scores
            scores_path = f"tmp/{split_type}.scores.html"
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
            violin_path = f"temp/{split}.violin.html"
            fig.write_html(violin_path)
            mlflow.log_artifact(violin_path)

    return

def stats_from_conf_mat(cm: ndarray):
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
