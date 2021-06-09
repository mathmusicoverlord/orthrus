"""
This file contains the experimental constants for the experiment setosa_versicolor_classify_species_svm.
All experimental parameters to be exported are denoted by UPPERCASE names as a convention.
"""

# imports
import datetime
import os
from datasci.core.dataset import load_dataset
from datasci.sparse.classifiers.svm import SSVMClassifier as SSVM
from datasci.sparse.classifiers.svm import L1SVM
from calcom.solvers import	LPPrimalDualPy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from ray import tune
import numpy as np

# set experiment name
EXP_NAME = 'setosa_versicolor_classify_species_svm'

# set working directories
PROJ_DIR = os.path.join(os.environ['DATASCI_PATH'], 'test_data', 'Iris') # <--- put your absolute path
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

# classification attribute
CLASS_ATTR = 'species'

# set classifier attributes
#CLASSIFIER = SSVM(solver=LPPrimalDualPy)
#CLASSIFIER_NAME = 'SSVM'
#CLASSIFIER_FWEIGHTS_HANDLE = 'weights_'
#kernel_args = dict(metric='linear')
kernel_args = None
CLASSIFIER = L1SVM(nu=1, delta=.001, imax=100, verbosity=0, kernel_args=kernel_args)
CLASSIFIER_NAME = 'l1SVM'
CLASSIFIER_FWEIGHTS_HANDLE = 'w_'
CLASSIFIER_SWEIGHTS_HANDLE = None
CLASSIFIER_TUNING_PARAMS = dict(nu=tune.grid_search(((2.0) ** np.arange(-12, 13)).tolist()),
                                delta=tune.grid_search(((10.0) ** np.arange(-3, 4)).tolist()))

# set 80/20 train/test split for simple experiment
#y = DATASET.metadata.loc[SAMPLE_IDS, CLASS_ATTR]
#split = train_test_split(y, test_size=.2)
#train_ids = split[0].index
#test_ids = split[1].index
#PARTITIONER = (train_ids, test_ids)
#PARTITIONER_NAME = 'split'

# maybe k-fold instead
PARTITIONER = KFold(n_splits=10, shuffle=True, random_state=0)
PARTITIONER_NAME = '10-fold'

# restrict features
FEATURE_IDS = None

# other parameters