"""
This module contains the classes and functions associated with pipeline components and workflows.
"""

from abc import ABC, abstractmethod
import os
import pandas as pd
from datasci.core.dataset import DataSet
from datasci.core.helper import generate_save_path
from datasci.core.helper import save_object
import warnings
from copy import deepcopy
import ray


class Process(ABC):

    def __init__(self,
                 process: object,
                 process_name: str = None,
                 parallel: bool = False,
                 verbosity: int = 0,
                 ):

        # set parameters
        self.process = process
        self.parallel = parallel
        self.verbosity = verbosity

        # set attributes
        self.run_status_ = 0
        self.results_ = None

        # private
        self._process_name = process_name

    @property
    def process_name(self):

        # set default name
        if self._process_name is None:
            self._process_name = self.process.__str__().replace('\n', '')

        return self._process_name

    def _preprocess(self, ds:DataSet, **kwargs):

        ## prep for subclasses _run method ##

        # grab transformation object
        transform = kwargs.get('transform', lambda x: x)
        ds_new = transform(ds)

        return ds_new

    @abstractmethod
    def _run(self, ds:DataSet, **kwargs):
        pass

    def run(self, ds:DataSet, batch_args:dict = None):

        if batch_args is None:
            self.results_ = dict(batch=self._run(ds))
            return

        # setup for sequential or parallel processesing
        if self.parallel:
            self.results_ = self._run_par(ds, batch_args)
        else:
            # intialize results
            self.results_ = dict()
            for batch in batch_args:
                if self.verbosity > 0:
                    print(batch + ':')
                self.results_[batch] = self._run(ds, **batch_args[batch])
                if self.verbosity > 0:
                    print()

        # change run status
        self.run_status_ = 1

        return ds, self.results_

    def _run_par(self, ds: DataSet, batch_args: dict):

        # define a remote _run method
        _run = ray.remote(lambda *args, **kwargs: self._run(*args, **kwargs))

        # store dataset in the object store
        if ds.__class__ == ray._raylet.ObjectRef:
            ds_remote = ds
        else:
            ds_remote = ray.put(ds)

        # naive ray parallelization
        futures = []
        for batch in batch_args:
            futures.append(_run.remote(ds_remote, **batch_args[batch]))

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
                 parallel: bool = False,
                 verbosity: int = 0,
                 split_attr: str = None,
                 split_group: str = None,
                 split_handle: str = 'split',
                 split_args: dict = {}):


        # init with Process class
        super(Partition, self).__init__(process=process,
                                        process_name=process_name,
                                        parallel=parallel,
                                        verbosity=verbosity,
                                        )

        # set parameters
        self.split_attr = split_attr
        self.split_group = split_group
        self.split_args = split_args
        self.split_handle = split_handle

    def _run(self, ds: DataSet, **kwargs):

        # run the supers _preprocess method
        ds_new = self._preprocess(ds, **kwargs)

        # check for validation in existing labels
        tvt_labels = kwargs.get('tvt_labels', None)

        if tvt_labels is None:
            tvt_labels = pd.Series(index=ds_new.metadata.index, data=['Train']*ds_new.n_samples)

        if (tvt_labels == 'Valid').any():
            raise ValueError("Labels already contain training, validation, and test!")

        # initialize parition result
        result = dict()

        # grab verbosity
        verbosity = self.verbosity

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
        if verbosity > 0:
            print(r"Generating %s splits..." % (self.process_name))

        parts = split(ds_new.data.loc[train_samples], **label_dict, **self.split_args)

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

        # change run status
        self.run_status_ = 1

        return ds, self.results_

    def save_results(self,
                    save_path: str,
                    overwrite: bool = False):

        # generate new save path in case you don't want to overwrite
        save_path = generate_save_path(save_path, overwrite)

        # generate dataframe to save labels
        results = pd.DataFrame.from_dict({v['tvt_labels'].name :v['tvt_labels'] for (k,v) in self.results_.items()})

        # save labels as .csv
        results.to_csv(save_path)


class Fit(Process):

    def __init__(self,
                 process: object,
                 process_name: str = None,
                 parallel: bool = False,
                 verbosity: int = 0,
                 supervised_attr: str = None,
                 fit_handle: str = 'fit',
                 fit_args: dict = {},
                 ):

        super(Fit, self).__init__(process=process,
                                  process_name=process_name,
                                  parallel=parallel,
                                  verbosity=verbosity,
                                  )

        # set parameters
        self.supervised_attr = supervised_attr
        self.fit_args = fit_args

        # set private attributes
        self._fit_handle = fit_handle

    def _extract_training_ids(self, ds: DataSet, **kwargs):

        # grab training labels
        tvt_labels = kwargs.get('tvt_labels', None)
        if tvt_labels is None:
            training_ids = ds.metadata.index
        else:
            training_ids = (tvt_labels == 'Train')

        return training_ids

    def _fit(self, ds: DataSet, **kwargs):

        # extract training ids
        training_ids = self._extract_training_ids(ds, **kwargs)

        #  add supervised labels to fit args
        if self.supervised_attr is not None:
            y = ds.metadata.loc[training_ids, self.supervised_attr]
            self.fit_args['y'] = y

        # fit the process
        if self.verbosity > 0:
            print(r"Fitting %s..." % (self.process_name,))
        process = deepcopy(self.process)
        process = eval("process." + self._fit_handle)(ds.data.loc[training_ids], **self.fit_args)

        return process


class Transform(Fit):

    def __init__(self,
                 process: object,
                 process_name: str = None,
                 parallel: bool = False,
                 verbosity: int = 0,
                 retain_f_ids: bool = False,
                 fit_handle: str = 'fit',
                 transform_handle: str = 'transform',
                 fit_transform_handle: str = 'fit_transform',
                 supervised_attr: str = None,
                 fit_args: dict = {}):


        # init with Process class
        super(Transform, self).__init__(process=process,
                                        process_name=process_name,
                                        parallel=parallel,
                                        verbosity=verbosity,
                                        supervised_attr=supervised_attr,
                                        fit_handle=fit_handle,
                                        fit_args=fit_args,
                                        )

        # set parameters
        self.retain_f_ids = retain_f_ids

        # set private attributes
        self._transform_handle = transform_handle
        self._fit_transform_handle = fit_transform_handle

        # check appropriate parameters
        if self._fit_handle is None or self._transform_handle is None:
            if self._fit_transform_handle is None:
                raise ValueError("Transform process must have either both a fit method and a transform method or just a fit_transform method!")
            else:
                warnings.warn("Transform will use its fit_transform method to fit."
                              " Make sure that its fit_transform method fits"
                              " the transformation inplace!")

    def _fit_transform(self, ds: DataSet, **kwargs):

        # extract training ids
        training_ids = self._extract_training_ids(ds, **kwargs)

        #  add supervised labels to fit args
        if self.supervised_attr is not None:
            y = ds.metadata.loc[training_ids, self.supervised_attr]
            self.fit_args['y'] = y

        # fit the process
        if self.verbosity > 0:
            print(r"Fitting %s and then transforming data..." % (self.process_name,))
        process = deepcopy(self.process)
        data = eval("process." + self._fit_transform_handle)(ds.data.loc[training_ids], **self.fit_args)

        return process, data

    def _run(self, ds:DataSet, **kwargs):

        # initalize output
        result = dict()

        # run the super's _preprocess method
        ds_new = self._preprocess(ds, **kwargs)

        # attempt to fit
        if self._transform_handle is not None:
            if self._fit_handle is not None:
                process = self._fit(ds_new, **kwargs)
            else:
                process, _ = self._fit_transform(ds_new, **kwargs)
        else:
            process = None

        # store resulting transform
        result['transform'] = self._generate_transform(process)

        return result

    def _generate_transform(self, process=None):

        # define transform
        def transform(ds: DataSet):
            if process is None:
                _, data_new = self._fit_transform(ds)
            else:
                if self.verbosity > 0:
                    print(r"Transforming the data using %s..." % (self.process_name,))
                data_new = eval("process." + self._transform_handle)(ds.data)
            if self.retain_f_ids:
                try:
                    data_new = pd.DataFrame(data=data_new, index=ds.data.index, columns=ds.data.columns)
                    ds_new = deepcopy(ds)
                    ds_new.data = data_new
                except ValueError:
                    raise ValueError("Transform changes the dimension of the data and therefore cannot retain the original feature ids in the new dataset!")
            else:
                data_new = pd.DataFrame(data=data_new, index=ds.data.index, columns=['_'.join([self.process_name, str(i)]) for i in range(data_new.shape[1])])

                # check if features are the same after transformation and use original feature ids for columns
                ds_new = DataSet(data=data_new, metadata=ds.metadata)

            return ds_new

        return transform

    def transform(self, ds: DataSet):

        # transform the incoming data according to transforms
        try:
            out = {k: v['transform'](ds) for (k, v) in self.results_.items()}
            return out
        except TypeError:
            raise RuntimeError("Transform must call its run method on a dataset before it can transform a new dataset!")

    def save_results(self, save_path: str, overwrite: bool = False):

        # call save_object on results
        save_object(self.results_, save_path, overwrite)


class Classify(Fit):

    def __init__(self,
                 process: object,
                 class_attr: str,
                 process_name: str = None,
                 parallel: bool = False,
                 verbosity: int = 0,
                 fit_handle: str = 'fit',
                 predict_handle: str = 'predict',
                 fit_args: dict = {},
                 predict_args: dict = {},
                 classes_handle='classes_',
                 ):

        # init with Process class
        super(Classify, self).__init__(process=process,
                                       process_name=process_name,
                                       parallel=parallel,
                                       verbosity=verbosity,
                                       supervised_attr=class_attr,
                                       fit_handle=fit_handle,
                                       fit_args=fit_args,
                                       )

        # set parameters
        self.fit_args = fit_args
        self.predict_args = predict_args
        self.class_attr = self.supervised_attr # shadows supervised_attr

        # set private attributes
        self._predict_handle = predict_handle
        self._classes_handle = classes_handle

        # check appropriate parameters
        if self._fit_handle is None or self._predict_handle is None:
            raise ValueError("Classify process must have both a fit method and a predict method!")


    def save_results(self, save_path: str, overwrite: bool = False):
        pass

    def _run(self, ds: DataSet, **kwargs):

        # initalize output
        result = dict()

        # run the super's _preprocess method
        ds_new = self._preprocess(ds, **kwargs)

        # fit the classifier
        process = self._fit(ds_new, **kwargs)

        # store resulting transform
        pred, pred_label = self._generate_labels_or_scores(process)
        result[pred_label] = pred

        return result

    def _generate_labels_or_scores(self, process: object, ds: DataSet):

        # predict on dataset
        if self.verbosity > 0:
            print(r"Classifying the data using %s..." % (self.process_name,))
        predictions = eval("process." + self._predict_handle)(ds.data)

        # format output as series if labels
        try:
            predictions.shape[1]
            pred = pd.DataFrame(data=predictions, index=ds.metadata.index, columns=eval("process." + self._classes_handle))
            pred.name = self.process_name + " scores"
            pred_label = 'class_scores'
        except IndexError:
            pred = pd.Series(name=self.process_name + " labels", data=predictions, index=ds.metadata.index)
            pred_label = 'class_labels'

        return pred, pred_label



