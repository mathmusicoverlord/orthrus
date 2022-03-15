"""Module for utility functions"""

# imports
from orthrus.core.pipeline import Report
from pandas import DataFrame
import mlflow
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
        fig.add_trace(go.Violin(y=score_frame[score].values, name=score, visible='legendonly'))
    
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
            violin_path = f"cache/{split_type}.violin.html"
            fig.write_html(violin_path)
            mlflow.log_artifact(violin_path)

    return