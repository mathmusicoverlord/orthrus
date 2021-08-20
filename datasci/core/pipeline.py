"""
This module contains the classes and functions associated with pipeline components and workflows.
"""

from abc import ABC, abstractmethod
from typing import Union
import os
import pandas as pd
from numpy import ndarray
import numpy as np
from pandas import DataFrame, Series
from datasci.core.dataset import DataSet
from datasci.core.helper import generate_save_path
from datasci.core.helper import save_object
import warnings
from copy import deepcopy
import ray

# module functions
def _compress_dict_dict_pandas(kv: dict, inner_key: str, **kwargs) -> DataFrame:

    # initalize out
    out = DataFrame()

    # grab the column suffix and prefix
    col_suffix = kwargs.get('col_suffix', None)
    col_suffix = '' if col_suffix is None else '_' + col_suffix
    col_prefix = kwargs.get('col_prefix', None)
    col_prefix = '' if col_prefix is None else col_prefix + '_'

    # infer Series or DataFrame
    pd_object = list(kv.values())[0][inner_key]

    # tuple comprehend the results based on type
    if type(pd_object) == Series:
        out = tuple(v[inner_key].rename(col_prefix + k + col_suffix) for (k, v) in kv.items())
    elif type(pd_object) == DataFrame:
        out = tuple(v[inner_key].rename(columns={col: col_prefix + '_'.join([k, col]) + col_suffix for col in v[inner_key].columns})
                    for (k, v) in kv.items())

    out = pd.concat(out, axis=1)
    out.index.name = kwargs.get('index_name', out.columns.name)
    out.columns.name = kwargs.get('columns_name', out.columns.name)

    return out


# module classes
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

    def compress_results(self, which: Union[list, str] = 'all') -> dict:
        # check that the process has been run
        assert self.results_ is not None, r"The %s process has not been run yet!" % (self.__class__.__name__,)

        # grab result keys
        keys = list(list(self.results_.values())[0].keys())

        # check if items in 'which' are in the keys
        which = keys if which == 'all' else ([which] if type(which) == str else which)
        assert set(which) <= set(keys), "Some or all of specified results do not exist in the keys!"

        # initialize output
        out = dict()

        # run the associated compression methods
        for key in which:
            try:
                result = eval("self._compress_" + key)()
                out.update({key: result})
            except AttributeError:
                warnings.warn("%s does not contain a compression method for %s! Returning \"uncompressed\" object." % (self.__class__.__name__, key))
                result = self._extract_result(key)

            # update dictionary
            out.update({key: result})

        return out

    def _extract_result(self, which: str):

        # check that the process has been run
        assert self.results_ is not None, r"The %s process has not been run yet!" % (self.__class__.__name__,)

        # grab result keys
        keys = list(list(self.results_.values())[0].keys())

        # check if items in 'which' are in the keys
        assert which in keys, "%s results does not exist in the keys!" % (which,)

        # initialize output
        out = dict()

        # extract object from each label
        for batch in self.results_:
            out.update({batch: self.results_[batch][which]})

        return out

    def save_results(self,
                     save_path: Union[str, dict] = None,
                     overwrite: bool = False,
                     compress: bool = True):

        # make sure there is either a save path or multiple
        assert save_path is not None, "You must provide a save path or multiple save paths in a dictionary!"

        # make sure the type is correct
        assert type(save_path) in [str, dict], "save_path must be a string or a dictionary with result keys and string values!"

        # acceptable formats
        formats = ['.csv', '.pickle']

        if type(save_path) == str:
            ext = '.' + save_path.split('.')[-1]
            assert ext == '.pickle', "The file extension must be .pickle for saving the results as a whole!"
            if compress:
                save_object(self.compress_results(), save_path, overwrite)
            else:
                save_object(self.results_, save_path, overwrite)

        else:
            assert compress, "Cannot save individual results without compressing the results first, set compress=True!"

            # compress the specificed results
            results = self.compress_results(which=list(save_path.keys()))

            # loop through each result and save
            for key in save_path.keys():
                # grab the specific result
                result = results[key]

                # check save path
                ind_save_path = generate_save_path(save_path[key], overwrite)

                # grab extension
                ext = '.' + ind_save_path.split('.')[-1]

                # check for appropriate extension
                assert ext in formats, "Saving of file extension %s not yet implemented, use one of the following: %s" \
                                       % (ext, ', '.join(formats))

                # saved base on extension type
                if ext == '.csv':
                    if type(result) == DataFrame or type(result) == Series:

                        # save pandas dataframe/series to .csv
                        result.to_csv(ind_save_path)
                        not_implemented_for_ext = False
                    elif type(result) is ndarray:

                        # save numpy array to .csv
                        np.savetxt(ind_save_path, result, delimiter=",")
                        not_implemented_for_ext = False
                    else:
                        not_implemented_for_ext = True

                elif ext == '.pickle':

                    # save with pickle
                    save_object(result, ind_save_path, overwrite)
                    not_implemented_for_ext = False

                if not_implemented_for_ext:
                    raise AttributeError("The compressed result %s may not be saved as %s!" % (key, ext))

        return

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

    def run(self, ds:DataSet, batch_args: dict = None, append_labels=True):

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

    def _compress_tvt_labels(self):

        # compress the dict of dict of series to dataframe
        results = _compress_dict_dict_pandas(kv=self.results_,
                                             inner_key="tvt_labels",
                                             columns_name=self.process_name + " splits",
                                             col_suffix="split")
        return results


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
                 fit_args: dict = {},
                 transform_args: dict = {}):


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
        self.transform_args = transform_args

        # set private attributes
        self._transform_handle = transform_handle
        self._fit_transform_handle = fit_transform_handle

        # check appropriate parameters
        if self._fit_handle is None or self._transform_handle is None:
            if self._fit_transform_handle is None:
                raise ValueError("%s process must have either both a fit method and a transform method or just a fit_transform method!" % (self.__class__.__name__,))
            else:
                warnings.warn("%s will use its fit_transform method to fit."
                              " Make sure that its fit_transform method fits"
                              " the transformation inplace!" % (self.__class__.__name__,))

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
        if self._fit_handle is not None:
            process = self._fit(ds_new, **kwargs)
        else:
            process, _ = self._fit_transform(ds_new, **kwargs)

        # store resulting transform
        result['transform'] = self._generate_transform(process)

        # store the resulting transformer
        result['transformer'] = process

        return result

    def _generate_transform(self, process=None):

        # define transform
        def transform(ds: DataSet):
            if self._transform_handle is None:
                _, data_new = self._fit_transform(ds)
            else:
                if self.verbosity > 0:
                    print(r"Transforming the data using %s..." % (self.process_name,))
                data_new = eval("process." + self._transform_handle)(ds.data, **self.transform_args)
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

        assert self.results_ is not None, "Transform must call its run method on a dataset before it can transform a new dataset!"

        # transform the incoming data according to transforms
        out = {k: v['transform'](ds) for (k, v) in self.results_.items()}
        return out


class FeatureSelect(Transform):

    def __init__(self,
                 process: object,
                 process_name: str = None,
                 parallel: bool = False,
                 verbosity: int = 0,
                 fit_handle: str = 'fit',
                 transform_handle: str = 'transform',
                 supervised_attr: str = None,
                 fit_args: dict = {},
                 transform_args: dict = {},
                 f_ranks_handle: str = None):


        # init with Process class
        super(FeatureSelect, self).__init__(process=process,
                                             process_name=process_name,
                                             parallel=parallel,
                                             verbosity=verbosity,
                                             supervised_attr=supervised_attr,
                                             fit_handle=fit_handle,
                                             fit_args=fit_args,
                                             transform_args=transform_args,
                                             transform_handle=transform_handle,
                                             )

        # remove unnecessary attribute
        del self.retain_f_ids

        # enforce there to be both a fit and a transform method
        assert self._fit_handle is not None and self._transform_handle is not None, \
            "Feature_Select process must have both a fit method for discovering" \
            " features and a transform method to restrict features."

        # private attributes
        self._f_ranks_handle = f_ranks_handle

    def _preprocess(self, ds:DataSet, **kwargs):
        return ds  # avoid double preprocessing from Transform

    def _run(self, ds: DataSet, **kwargs):

        # preprocess data with the super
        ds_new = super(FeatureSelect, self)._preprocess(ds, **kwargs)

        # run the super method
        result = super(FeatureSelect, self)._run(ds_new, **kwargs)

        # change name of transformer to selector
        process = result['transformer']
        result['selector'] = process
        del result['transformer']

        # append feature ranks
        result.update(self._generate_f_ranks(process, ds_new))

        return result

    def _generate_transform(self, process=None):

        # define transform
        def transform(ds: DataSet):
            if self.verbosity > 0:
                print(r"Restricting features in the data using %s..." % (self.process_name,))
            select = eval("process." + self._transform_handle)
            feature_ids = select(ds.data.columns.to_numpy().reshape(1, -1), **self.transform_args)
            ds_new = ds.slice_dataset(feature_ids=feature_ids)

            return ds_new

        return transform

    def _generate_f_ranks(self, process: object, ds: DataSet):

        # initialize out
        out = {}

        # check for f_ranks attribute
        if self._f_ranks_handle is not None:
            f_ranks = eval("process." + self._f_ranks_handle)
            if type(f_ranks) == DataFrame:
                f_ranks.rename(index=dict(zip(f_ranks.index.tolist(), ds.vardata.index.tolist())))
                f_ranks.columns.name = self.process_name + " f_ranks"
            else:
                f_ranks = np.array(f_ranks)  # convert to ndarray

                # check shape
                if len(f_ranks.shape) == 1:
                    f_ranks = Series(index=ds.vardata.index,
                                     data=f_ranks,
                                     name=self.process_name + " f_ranks")
                else:
                    f_ranks = DataFrame(index=ds.vardata.index,
                                        data=f_ranks)
                    f_ranks.columns.name = self.process_name + " f_ranks"

            # update output
            out.update({'f_ranks': f_ranks})

        return out

    def _compress_f_ranks(self):

        # compress the dict of dict of dataframe to dataframe
        results = _compress_dict_dict_pandas(kv=self.results_,
                                             inner_key="f_ranks",
                                             columns_name=self.process_name + " f_ranks",
                                             col_suffix="f_ranks")
        return results

    def transform(self, ds: DataSet):

        assert self.results_ is not None, "Transform must call its run method on a dataset before it can transform a new dataset!"

        # transform the incoming data according to transforms
        out = {k: v['transform'](ds) for (k, v) in self.results_.items()}
        return out


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
                 f_weights_handle: str = None,
                 s_weights_handle: str = None,
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
        self._f_weights_handle = f_weights_handle
        self._s_weights_handle = s_weights_handle

        # check appropriate parameters
        if self._fit_handle is None or self._predict_handle is None:
            raise ValueError("Classify process must have both a fit method and a predict method!")

    def _compress_class_labels(self):

        # compress the dict of dict of series to dataframe
        results = _compress_dict_dict_pandas(kv=self.results_,
                                             inner_key="class_labels",
                                             columns_name=self.process_name + " labels",
                                             col_suffix='_'.join([self.process_name, "labels"]))
        return results

    def _compress_class_scores(self):

        # compress the dict of dict of dataframe to dataframe
        results = _compress_dict_dict_pandas(kv=self.results_,
                                             inner_key="class_labels",
                                             which='dataframe',
                                             columns_name=self.process_name + " scores",
                                             col_suffix="scores")
        return results

    def _compress_f_weights(self):

        # compress the dict of dict of series to dataframe
        results = _compress_dict_dict_pandas(kv=self.results_,
                                             inner_key="f_weights",
                                             columns_name=self.process_name + " f_weights",
                                             col_suffix="f_weights")

        return results

    def _compress_s_weights(self):

        # compress the dict of dict of series to dataframe
        results = _compress_dict_dict_pandas(kv=self.results_,
                                             inner_key="s_weights",
                                             columns_name=self.process_name + " s_weights",
                                             col_suffix="s_weights")

        return results

    def _run(self, ds: DataSet, **kwargs):

        # initialize output
        result = dict()

        # run the super's _preprocess method
        ds_new = self._preprocess(ds, **kwargs)

        # fit the classifier
        process = self._fit(ds_new, **kwargs)

        # store resulting transform
        result.update(self._generate_labels_or_scores(process, ds_new))

        # grab potential feature and sample weights
        result.update(self._generate_f_s_weights(process, ds_new))

        # store the fit process
        result.update({'classifier': process})

        return result

    def _generate_labels_or_scores(self, process: object, ds: DataSet) -> dict:

        # predict on dataset
        if self.verbosity > 0:
            print(r"Classifying the data using %s..." % (self.process_name,))
        predictions = eval("process." + self._predict_handle)(ds.data)

        # format output as series if labels
        if len(predictions.shape) == 2:
            pred = pd.DataFrame(data=predictions, index=ds.metadata.index, columns=eval("process." + self._classes_handle))
            pred.columns.name = self.process_name + " scores"
            pred_label = 'class_scores'
        else:
            pred = pd.Series(name=self.process_name + " labels", data=predictions, index=ds.metadata.index)
            pred_label = 'class_labels'

        return {pred_label: pred}

    def _generate_f_s_weights(self, process: object, ds: DataSet) -> dict:

        # initialize output
        out = dict()

        # extract feature weights
        if self._f_weights_handle is not None:
            f_weights = pd.Series(index=ds.vardata.index,
                                  data=eval("process." + self._f_weights_handle),
                                  name='_'.join([self.process_name, 'f_weights']))

            out.update({'f_weights': f_weights})

        # extract sample weights
        if self._s_weights_handle is not None:
            s_weights = pd.Series(index=ds.metadata.index,
                                  data=eval("process." + self._s_weights_handle),
                                  name='_'.join([self.process_name, 's_weights']))

            out.update({'s_weights': s_weights})


        return out


