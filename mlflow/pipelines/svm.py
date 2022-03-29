"""This is an example pipeline for classification with svm."""

# imports
import os
from statistics import mode
import sys
import ray.tune
from pandas import DataFrame
from orthrus.core.pipeline import *
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score

# set tuning parameters
def config() -> dict:
    return {'svm_C': ray.tune.loguniform(1e-4, 1e2)}

def searcher() -> ray.tune.suggest.suggestion.Searcher:
    return HyperOptSearch(metric='mean_valid_bsr', mode='max')

def search_alg() -> ray.tune.suggest.suggestion.Searcher:
     return ConcurrencyLimiter(searcher(), max_concurrent=4)

def scheduler() -> ray.tune.schedulers.trial_scheduler.TrialScheduler:
     return AsyncHyperBandScheduler(metric='mean_valid_bsr', mode='max')

# set the score function
def score(pipeline: Pipeline) -> dict:
    
    # extract mean bsr across folds
    report: Report = pipeline.processes[-1]
    scores: DataFrame = report.report()['train_valid_test']
    mean_bsr = scores['Valid:macro avg:Recall'].mean()

    return {'mean_valid_bsr': mean_bsr}

# set final hyperparamters
def hyperparams() -> dict():
    return {'svm_C': 0.0033588966840458474}

# build pipeline
def generate_pipeline(svm_C: float = 1, **kwargs) -> Pipeline:

    # set train/test partitioner
    shuffle = Partition(process=ShuffleSplit(n_splits=1, test_size=.2, random_state=8756),
                        process_name='shuffle')

    # set validation partitioner
    kfold = Partition(process=KFold(n_splits=5, shuffle=True, random_state=443),
                     process_name='5fold')

    # standardize
    std = Transform(process=StandardScaler(),
                    process_name='std',
                    retain_f_ids=True)
    
    # support vector machines
    svm = Classify(process=LinearSVC(C=svm_C),
                   process_name='svm',
                   class_attr='Shedding',
                   )
    
    # score
    # bsr = Score(process=balanced_accuracy_score,
    #             process_name='bsr',
    #             pred_attr='Shedding')

    # report
    report = Report(pred_attr='Shedding')
    
    # grab checkpoint path
    checkpoint_path = kwargs.get('checkpoint_path', None)

    # create pipeline
    pipeline = Pipeline(processes=(shuffle, kfold, std, svm, report),
                        pipeline_name='svm_classify',
                        checkpoint_path=checkpoint_path,
                        )

    return pipeline

