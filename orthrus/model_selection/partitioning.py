'''This module defines various partitioning classes which partitions samples in batches of training and test split. 
These classes must have the sklearn equivalent of a split method. The split method returns a list of train-test partitions; 
one for each fold in the experiment. See sklearn.model_selection.KFold for an example partitioner.'''

# imports
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
import copy 
import pandas as pd

class TrainTestPartitioner():
    def __init__(self, train_idx, test_idx):
        self.train_idx = train_idx
        self.test_idx = test_idx

    def split(self, X=None, y=None):
        return [[self.train_idx, self.test_idx]]

class TrainPartitioner():
    def __init__(self):
        pass
    def split(self, X=None, y=None):
        return [[np.arange(np.shape(X)[0]), []]]


class StratifiedGroupKFoldForEachUniqueY():
        def __init__(self, n_splits, shuffle=False, random_state=42):
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state

        def __str__(self):
            return "StratifiedGroupKFoldForEachUniqueY(n_splits=%d, shuffle=%r, random_state=%d"%(self.n_splits, self.shuffle, self.random_state)
        
        def split(self, X, y, groups):
                train = {}
                test = {}
                if isinstance(X, pd.core.frame.DataFrame):
                    X = X.values
                if isinstance(y, pd.core.frame.DataFrame):
                    y = y.values
                if isinstance(groups, pd.core.frame.DataFrame):
                    groups = groups.values

                for unique_y in np.unique(y):
                        idxs_unique_y = np.where(y == unique_y)[0]
                        cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
                        splits = cv.split(X[idxs_unique_y], y[idxs_unique_y], groups[idxs_unique_y])
                        for i, split in enumerate(splits):
                                train_idxs = train.get(i, [])
                                test_idxs = test.get(i, [])
                                
                                train_idxs.extend(idxs_unique_y[split[0]])
                                test_idxs.extend(idxs_unique_y[split[1]])

                                train[i] = train_idxs
                                test[i] = test_idxs
                
                self.splits = []
                for k, train_list in train.items():
                        test_list = test[k]
                        temp = copy.copy(train_list)
                        temp.extend(test_list)
                        assert np.unique(temp).shape[0] == X.shape[0], 'All samples not present in the split'
                        self.splits.append([train_list, test_list])

                return self.splits


class TrainValidationTestPartitioner():
    '''
    NOTE: this class is meant to be used with orthrus.core.pipeline.Pipeline only. 
    The split method of this class returns a list of train-validation partitions, but the Pipeline
    framework will automatically create test partition by using the remaining samples as the test partition.

    This class creates train-validation-test partitions using the following scheme:
    1. Partition the data into train-test sets using the partitioner provided in the constructor
    2. For each train partition from the previous step: create train-validation set using StratifiedShuffleSplit

    parameters:
        partitioner - partitioner to use for creating train-test partitions
        validation_set_size - fraction of samples to use for validation set
        random_state - random state to use for StratifiedShuffleSplit

    returns:
        train-validation-test partitions
    '''
    def __init__(self, outer_partitioner, 
                        inner_partitioner, 
                        random_inner_partition=False, 
                        random_state=42, 
                        returns=['train', 'validation']):

                        
        self.outer_partitioner = outer_partitioner
        self.inner_partitioner = inner_partitioner        
        self.random_inner_partition = random_inner_partition
        self.random_state = random_state
        self.returns = [x.lower() for x in returns]

    def split(self, x, y, groups=None):
        np.random.seed(self.random_state)
        partitions = self.outer_partitioner.split(x, y, groups)
        for intermediate_partition, test in partitions:
            if 'train' not in self.returns and 'validation' not in self.returns:
                yield test

            intermediate_partition = np.array(intermediate_partition)
            if groups is None:
                splits = list(self.inner_partitioner.split(x.iloc[intermediate_partition], y[intermediate_partition]))
            else:
                splits = list(self.inner_partitioner.split(x.iloc[intermediate_partition], y[intermediate_partition], groups[intermediate_partition]))

            if len(splits) > 1:
                if self.random_inner_partition == False:
                    inner_partition_number = 0
                else:
                    
                    inner_partition_number = np.random.randint(0, len(splits))
          
                train =  intermediate_partition[splits[inner_partition_number][0]]
                validation = intermediate_partition[[splits[inner_partition_number][1]]]
            else:
                train =  intermediate_partition[splits[0]]
                validation = intermediate_partition[[splits[1]]]


            rets = []
            if 'train' in self.returns:
                rets.append(train)
            if 'validation' in self.returns:
                rets.append(validation)
            if 'test' in self.returns:
                rets.append(test)
            
            yield rets