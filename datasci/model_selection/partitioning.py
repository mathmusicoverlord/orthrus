'''This module defines various partitioning classes which partitions samples in batches of training and test split. 
These classes must have the sklearn equivalent of a split method. The split method returns a list of train-test partitions; 
one for each fold in the experiment. See sklearn.model_selection.KFold for an example partitioner.'''

class TrainTestPartitioner():
    def __init__(self, train_idx, test_idx):
        self.train_idx = train_idx
        self.test_idx = test_idx

    def split(self, X=None, y=None):
        return [[self.train_idx, self.test_idx]]