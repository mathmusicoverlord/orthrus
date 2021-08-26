"""
This file contains the experimental constants for the experiment classify_setosa_versicolor_svm.
All experimental parameters to be exported are denoted by UPPERCASE names as a convention.
"""

# imports
import datetime
import os
from orthrus.core.dataset import load_dataset
from orthrus.sparse.classifiers.svm import SSVMClassifier as SSVM
from orthrus.sparse.classifiers.svm import L1SVM
from calcom.solvers import LPPrimalDualPy
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from ray import tune
import numpy as np

# set experiment name
EXP_NAME = 'classify_setosa_versicolor_svm'

# set working directories
PROJ_DIR = os.path.join(os.environ['ORTHRUS_PATH'], 'test_data', 'Iris') # <--- put your absolute path
DATA_DIR = os.path.join(PROJ_DIR, 'Data')
EXP_DIR = os.path.join(PROJ_DIR, 'Experiments', EXP_NAME)
RESULTS_DIR = os.path.join(EXP_DIR, 'Results')

# generate figures directory by date
dt = datetime.datetime.now()
dt = datetime.date(dt.year, dt.month, dt.day)
FIG_DIR = os.path.join(EXP_DIR, 'Figures', dt.__str__())
os.makedirs(FIG_DIR, exist_ok=True)

# load dataset
DATASET = load_dataset(os.path.join(DATA_DIR, 'iris.ds'))
DATASET.path = FIG_DIR

# restrict samples
SAMPLE_IDS = DATASET.metadata['species'].isin(['setosa', 'versicolor'])

# restrict features
FEATURE_IDS = None

# classification attribute
CLASS_ATTR = 'species'

## specific script args

# classify.py args
CLASSIFY_ARGS = dict(PARTITIONER=ShuffleSplit(n_splits=1, train_size=.8),
                     CLASSIFIER=SSVM(C=1, use_cuda=True, solver=LPPrimalDualPy),
                     CLASSIFIER_NAME='SSVM',
                     CLASSIFIER_F_WEIGHTS_HANDLE='weights_',
                     CLASSIFIER_S_WEIGHTS_HANDLE=None,
                     )

# visualize.py args
VISUALIZE_ARGS = dict()