"""This is an example pipeline for classification with svm."""

# imports
import os
from re import M
from statistics import mode
import sys
import ray.tune
from pandas import DataFrame
from orthrus.core.pipeline import *
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import balanced_accuracy_score

# set tuning parameters
def config() -> dict:
    return {'rfc_min_impurity_decrease': ray.tune.uniform(0, 1),
            'rfc_n_estimators': ray.tune.choice(np.arange(20, 220, 20))}

def searcher() -> ray.tune.suggest.suggestion.Searcher:
    return HyperOptSearch(metric='mean_valid_f1', mode='max')

def search_alg() -> ray.tune.suggest.suggestion.Searcher:
     return ConcurrencyLimiter(searcher(), max_concurrent=4)

def scheduler() -> ray.tune.schedulers.trial_scheduler.TrialScheduler:
     return AsyncHyperBandScheduler(metric='mean_valid_f1', mode='max')

# set the score function
def score(pipeline: Pipeline) -> dict:
    
    # extract mean bsr across folds
    report: Report = pipeline.processes[-1]
    scores: DataFrame = report.report()['train_valid_test']
    mean_f1 = scores['Valid:weighted avg:F1-score'].mean()

    return {'mean_valid_f1': mean_f1}

# set final hyperparamters
def hyperparams() -> dict():
    return {'rfc_min_impurity_decrease': 0.0025117735639698657,
            'rfc_n_estimators': 120}

# build pipeline
def generate_pipeline(rfc_n_estimators, rfc_min_impurity_decrease, **kwargs) -> Pipeline:

    # generate train/test partition
    train_test = Partition(process=StratifiedShuffleSplit(n_splits=1,
                                                          test_size=.2,
                                                          random_state=983475),
                           process_name='train_test',
                           split_attr='SubjectID',
                           )

    # generate train/test partition
    train_valid_test = Partition(process=StratifiedShuffleSplit(n_splits=1,
                                                                test_size=.2,
                                                                random_state=568),
                                 process_name='train_valid_test',
                                 split_attr='SubjectID',
                                 )

    # standardize
    std = Transform(process=StandardScaler(),
                    process_name='std',
                    retain_f_ids=True,
                    )
    
    # generate RFC process
    rfc = Classify(process=RFC(n_estimators=rfc_n_estimators,
                               min_impurity_decrease=rfc_min_impurity_decrease,
                               ),
                   process_name='rfc',
                   class_attr='Shedding')

    # report
    report = Report(pred_attr='Shedding')
    
    # grab checkpoint path
    checkpoint_path = kwargs.get('checkpoint_path', None)

    # create pipeline
    pipeline = Pipeline(processes=(train_test, train_valid_test, std, rfc, report),
                        pipeline_name='rfc_classify',
                        checkpoint_path=checkpoint_path,
                        )

    return pipeline

