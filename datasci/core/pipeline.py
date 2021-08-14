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
                 process:object,
                 process_name:str = None,
                 parallel:bool = False
                 ):

        # set parameters
        self.process = process
        self._process_name = process_name
        self.parallel = parallel

        # set attributes
        self.run_status_ = -1
        self.results_ = None

    @property
    def process_name(self):

        # set default name
        if self._process_name is None:
            self._process_name = self.process.__str__().replace('\n', '')

        return self._process_name

    def run_(self, ds:DataSet, **kwargs):

        ## prep for subclasses run_ method ##

        # grab transformation object
        transform = kwargs.get('transform', lambda x: x)
        ds_new = transform(ds)

        return ds_new

    def run(self, ds:DataSet, batch_args:dict = None):

        if batch_args is None:
            self.results_ = dict(batch=self.run_(ds))
            return

        # setup for sequential or parallel processesing
        if self.parallel:
            self.results_ = self.run_par_(ds, batch_args)
        else:
            # intialize results
            self.results_ = dict()
            for batch in batch_args:
                self.results_[batch] = self.run_(ds, **batch_args[batch])

    def run_par_(self, ds:DataSet, batch_args: dict):

        # define a remote run_ method
        run_ = ray.remote(self.run_)

        # store dataset in the object store
        if ds.__class__ == ray._raylet.ObjectRef:
            ds_remote = ds
        else:
            ds_remote = ray.put(ds)

        # naive ray parallelization
        futures = []
        for batch in batch_args:
            futures.append(run_.remote(ds_remote, **batch_args[batch]))

        # collect results
        return dict(zip(batch_args.keys(), ray.get(futures)))

    @abstractmethod
    def save_results(self,
             save_path: str,
             overwrite: bool = False):
        pass

    def save(self,
             save_path: str,
             overwrite: bool = False):

        # generate new save path in case you don't want to overwrite
        save_path = generate_save_path(save_path, overwrite)

        # save instance of object
        save_object(self, save_path)




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

    def run_(self, ds:DataSet, **kwargs):

        # run the supers run_ method
        ds_new = super(Partition, self).run_(ds, **kwargs)

        # check for validation in existing labels
        tvt_labels = kwargs.get('tvt_labels', None)

        if tvt_labels is None:
            tvt_labels = pd.Series(index=ds_new.metadata.index, data=['Train']*ds_new.n_samples)

        if (tvt_labels == 'Valid').any():
            raise ValueError("Labels already contain training, validation, and test!")

        # initialize parition result
        result =  dict()

        # generate the split method
        split = eval("self.process." + self.split_handle)

        # grab training labels
        train_samples = tvt_labels[tvt_labels == 'Train'].index

        # generate labels
        if self.split_attr is None:
            label_dict = {}
        else:
            label_dict = dict(y=ds_new.metadata.loc[train_samples, self.split_attr])

        if self.split_group is not None:
            label_dict['groups'] = ds_new.metadata.loc[train_samples, self.split_group]

        # partition the dataset
        parts = split(ds_new.data, **label_dict, **self.split_args)

        # record training and test labels
        train_test_labels = pd.DataFrame(index=ds_new.metadata.index)
        for i, (train_idx, test_idx) in enumerate(parts):

            # create new column for split
            col_name = i
            train_test_labels[col_name] = 'Test'

            # fill in training and test
            train_test_labels.loc[train_samples[train_idx], i] = 'Train'
            train_test_labels.loc[train_samples[test_idx], i] = 'Valid'

        # check rename valid to test if there are no test left
        if (train_test_labels != 'Test').all().all():
            train_test_labels.replace('Valid', 'Test', inplace=True)

        # store the result
        result['tvt_labels'] = train_test_labels

        # return to run method
        return result

    def run(self, ds:DataSet, batch_args:dict = None, append_labels=True):

        # run the super first
        super(Partition, self).run(ds, batch_args)

        # collect super results
        results = dict()

        # split into batches
        for batch in self.results_:
            labels = self.results_[batch]['tvt_labels']
            labels.columns = ['_'.join([str(batch), str(col)])  for col in labels]
            try:
                orig_name = batch_args[batch]['tvt_labels'].name
            except (TypeError, KeyError):
                orig_name = ''
            labels = labels.to_dict('series')
            labels = {k: dict(tvt_labels=v.rename('_'.join([orig_name, self.process_name, str(i)]).lstrip('_'))) for i, (k,v) in enumerate(labels.items())}
            results.update(labels)

        # append original labels
        if append_labels and (batch_args is not None):
            results.update({k: dict(tvt_labels=v['tvt_labels']) for (k,v) in batch_args.items()})
            try:
                del results['batch']
            except KeyError:
                pass

        # update results
        self.results_ = results






    def save_results(self,
                    save_path: str,
                    overwrite: bool = False):

        # generate new save path in case you don't want to overwrite
        save_path = generate_save_path(save_path, overwrite)

        # save labels as .csv
        self.results_.to_csv(save_path)










