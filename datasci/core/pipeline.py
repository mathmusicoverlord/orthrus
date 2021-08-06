"""
This module contains the classes and functions associated with pipeline components and workflows.
"""

from abc import ABC, abstractmethod
import os
import pandas as pd
from datasci.core.dataset import DataSet
from datasci.core.helper import generate_save_path
from datasci.core.helper import save_object


class Process(ABC):

    def __init__(self,
                 process: object,
                 process_name: str = None,
                 ):

        # set parameters
        self.process = process
        self._process_name = process_name

        # set attributes
        self.run_status_ = -1
        self.result_ = None

    @property
    def process_name(self):

        # set default name
        if self._process_name is None:
            self._process_name = self.process.__str__().replace('\n', '')

        return self._process_name

    @abstractmethod
    def run(self, ds: DataSet, **kwargs):
        pass

    @abstractmethod
    def save(self,
             save_path: str,
             overwrite: bool = False):
        pass





class Partition(Process):

    def __init__(self,
                 process: object,
                 process_name: str = None,
                 split_attr: str = None,
                 split_group: str = None,
                 split_handle: str = 'split',
                 split_args: dict = {}):


        # init with Process class
        super(Partition, self).__init__(process=process,
                                        process_name=process_name,
                                        )

        # set parameters
        self.split_attr = split_attr
        self.split_group = split_group
        self.split_args = split_args
        self.split_handle = split_handle

        # set attributes

    def run(self, ds: DataSet, **kwargs):

        # generate the split method
        split = eval("self.process." + self.split_handle)

        # generate labels
        if self.split_attr is None:
            label_dict = {}
        else:
            label_dict = dict(y=ds.metadata[self.split_attr])

        if self.split_group is not None:
            label_dict['groups'] = self.split_group

        # partition the dataset
        parts = split(ds.data, **label_dict, **self.split_args)

        # record training and test labels
        train_test_labels = pd.DataFrame(index=ds.metadata.index)
        for i, (train_idx, test_idx) in enumerate(parts):

            # create new column for split
            col_name = 'split_' + str(i)
            train_test_labels[col_name] = ''

            # fill in training and test
            train_test_labels.iloc[train_idx, i] = 'Train'
            train_test_labels.iloc[test_idx, i] = 'Test'

        # store the result
        self.result_ = train_test_labels

        return ds

    def save(self,
             save_path: str,
             overwrite: bool = False):

        # generate new save path in case you don't want to overwrite
        save_path = generate_save_path(save_path, overwrite)

        # save labels as .csv
        self.result_.to_csv(save_path)










