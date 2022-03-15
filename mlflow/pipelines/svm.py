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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score

# set tuning parameters
def config() -> dict:
    return {'svm_C': ray.tune.loguniform(1e-4, 1e2)}

def searcher() -> ray.tune.suggest.suggestion.Searcher:
    return HyperOptSearch(metric='score', mode='max')

def search_alg() -> ray.tune.suggest.suggestion.Searcher:
     return ConcurrencyLimiter(searcher(), max_concurrent=4)

def scheduler() -> ray.tune.schedulers.trial_scheduler.TrialScheduler:
     return AsyncHyperBandScheduler(metric='score', mode='max')

# set the score function
def score(pipeline: Pipeline) -> float:
    
    # extract mean bsr across folds
    bsr: Score = pipeline.processes[-1]
    scores: DataFrame = bsr.collapse_results()['class_pred_scores']
    mean_bsr = scores.loc['Test'].mean()

    return mean_bsr


# set final hyperparamters
def hyperparams() -> dict():
    return {'svm_C': 1}

# build pipeline
def generate_pipeline(svm_C: float = 1, **kwargs) -> Pipeline:

    # set partitioner
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
    bsr = Score(process=balanced_accuracy_score,
                process_name='bsr',
                pred_attr='Shedding')

    # clear checkpoint path
    #checkpoint_path = os.path.join(os.environ['ORTHRUS_PATH'], "mlflow/tmp/svm.pickle")
    #os.system(f"rm -f {checkpoint_path}")
    
    # create pipeline
    pipeline = Pipeline(processes=(kfold, std, svm, bsr),
                        pipeline_name='svm_classify',
                        #checkpoint_path=checkpoint_path,
                        )

    return pipeline

