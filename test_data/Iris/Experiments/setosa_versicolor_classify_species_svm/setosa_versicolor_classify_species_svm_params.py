"""
This file contains the experimental constants for the experiment setosa_versicolor_classify_species_svm.
All experimental parameters to be exported are denoted by UPPERCASE names as a convention.
"""

# imports
import datetime
import os
from datasci.core.dataset import load_dataset
from datasci.sparse.classifiers.svm import SSVMClassifier as SSVM
from calcom.solvers import	LPPrimalDualPy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# set experiment name
EXP_NAME = 'setosa_versicolor_classify_species_svm'

# set working directories
PROJ_DIR = os.getcwd().replace('\\','/').replace('Scripts', '').rstrip('/') + '/' # <--- put your absolute path
DATA_DIR = PROJ_DIR + 'Data/'
EXP_DIR = PROJ_DIR + 'Experiments/' + EXP_NAME + '/'
RESULTS_DIR = EXP_DIR + 'Results/'

# generate figures directory by date
dt = datetime.datetime.now()
dt = datetime.date(dt.year, dt.month, dt.day)
FIG_DIR = EXP_DIR + 'Figures/' + dt.__str__() + '/'
try:
	os.mkdir(FIG_DIR.rstrip('/'))
except Exception:
	pass

# load dataset
DATASET = load_dataset(DATA_DIR + 'iris.ds')
DATASET.path = FIG_DIR

# restrict samples
SAMPLE_IDS = DATASET.metadata['species'].isin(['setosa', 'versicolor'])

# classification attribute
CLASS_ATTR = 'species'

# set classifier attributes
CLASSIFIER = SSVM(solver=LPPrimalDualPy)
CLASSIFIER_NAME = 'SSVM'
CLASSIFIER_WEIGHTS_HANDLE = 'weights_'

# set 80/20 train/test split for simple experiment
#y = DATASET.metadata.loc[SAMPLE_IDS, CLASS_ATTR]
#split = train_test_split(y, test_size=.2)
#train_ids = split[0].index
#test_ids = split[1].index
#PARTITIONER = (train_ids, test_ids)
#PARTITIONER_NAME = 'split'

# maybe k-fold instead
PARTITIONER = KFold(n_splits=5, shuffle=True, random_state=0)
PARTITIONER_NAME = '5-fold'

# restrict features
#FEATURE_IDS = DATASET.vardata.query('query here')

# other parameters