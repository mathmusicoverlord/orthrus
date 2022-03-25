"""
This module contains the classes and functions associated with process and pipeline components.
"""

from abc import ABC, abstractmethod
from typing import Union, Callable, Tuple
import os
import inspect
import pandas as pd
import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from orthrus.core.dataset import DataSet
from orthrus.core.helper import generate_save_path
from orthrus.core.helper import save_object, load_object
from sklearn.metrics import classification_report
import warnings
from copy import copy, deepcopy
import ray


# module functions
def _valid_args(func: Callable) -> list:
    """
    This function takes a function and returns the keyword arguments that are valid for input.

    Args:
        func (Callable): The function whose arguments are to be inspected.

    Returns:
        list : Contains the keyword arguments which are valid for ``func``.
    """

    return list(inspect.signature(func).parameters.keys())


def _collapse_dict_dict_pandas(kv: dict, inner_key: str, **kwargs) -> DataFrame:
    """
    Takes a dictionary :py:attr:`kv` of dictionaries ``kvi``, each which contains pandas
    ``Series`` or ``DataFrame`` as values,
    extracts the pandas object from each inner dictionary ``kvi`` at the specified :py:attr:`inner_key`,
    and then compiles them into a single ``DataFrame`` object.

    Args:
        kv (dict): Dictionary of dictionaries

        inner_key (str): Key to use on the inner dictionaries where the ``Series`` or ``DataFrame`` values exist.

        **kwargs: Additional arguments for specifying the column names, index name, and column prefix and suffix, for
            the output ``DataFrame``.

    Returns:
        DataFrame : The concatenated ``DataFrame`` or ``Series`` objects contained within the inner dictionaries.
    """

    # initalize out
    out = DataFrame()

    # grab the column suffix and prefix
    col_suffix = kwargs.get('col_suffix', None)
    col_suffix = '' if col_suffix is None else f"_{col_suffix}"
    col_prefix = kwargs.get('col_prefix', None)
    col_prefix = '' if col_prefix is None else f"{col_prefix}_"

    # infer Series or DataFrame
    pd_object = list(kv.values())[0][inner_key]

    # tuple comprehend the results based on type
    if type(pd_object) == Series:
        out = tuple(v[inner_key].rename(f"{col_prefix}{k}{col_suffix}") for (k, v) in kv.items())
    elif type(pd_object) == DataFrame:
        out = tuple(v[inner_key].rename(
            columns={col: f"{col_prefix}{k}:{col}{col_suffix}" for col in v[inner_key].columns}) for (k, v) in
                    kv.items())

    out = pd.concat(out, axis=1)
    out.index.name = kwargs.get('index_name', out.index.name)
    out.columns.name = kwargs.get('columns_name', out.columns.name)

    return out


def compose(funcs: Tuple[Callable, ...]):
    """
    This function takes a tuple of functions :math:`(f_1,\ldots,f_n)` and returns their composition
    :math:`f = f_1\circ\cdots\circ f_n`.

    Args:
        funcs (tuple of Callable): Tuple of functions to be composed.

    Returns:
        Callable : Composition of the above tuple of functions.
    """
    # define the composition
    def f(x: object):

        # set initial output
        y = deepcopy(x)

        # apply each function iteratively
        for i, fi in enumerate(funcs):
            if fi is not None:
                y = fi(y)
        return y

    return f


# module classes
class Process(ABC):
    """
    The base class for all processes in the pipeline module. Processes wrap class instances and functions for machine
    learning task, e.g, normalization via an object with a ``fit`` and ``transform`` method, classification via an
    object with a ``fit`` and ``predict`` method, etc... Fits well within the
    `scikit-learn API <https://scikit-learn.org/stable/>`_, but can be adapted to other popular machine learning
    libraries. A :py:class:`Process` instance is meant to be run on a :py:class:`DataSet` instance, via the method
    :py:meth:`Process.run`.

    Parameters:
        process (object or Callable): The object to perform the action defined by the :py:class:`Process` instance.

        process_name (str): The common name assigned to the :py:attr:`process`.

        parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
            Default is False. :py:func:`ray.init` must be called to initiate the ray cluster before any running
            can be done.

        verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
            the verbosity the larger the text output. Default is 1, indicating the standard text output with a
            :py:class:`Process` instance.

    Attributes:
        process (object or Callable): The object to perform the action defined by the :py:class:`Process` instance.

        parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
            Default is False:

        verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
            the verbosity the larger the text output. Default is 1, indicating the standard text output with a
            :py:class:`Process` instance.

        run_status_ (int): Indicates whether or not the process has finished. A value of 0 indicates the process has not
            finished, a value of 1 indicated the process has finished.

        results_ (dict of dicts): The results of the run process. The keys of the dictionary indicates the batch results
            for a given batch. For each batch there is a dictionary of results with keys indicating the result type, e.g,
            training/validation/test labels (``tvt_labels``), classification labels (``class_labels``), etc...
    """

    def __init__(self,
                 process: object,
                 process_name: str = None,
                 parallel: bool = False,
                 verbosity: int = 1,
                 ):

        # set parameters
        self.process = process
        self.parallel = parallel
        self.verbosity = verbosity

        # set attributes
        self.run_status_ = 0
        self.results_ = None
        self._labels: list = ["process", "process_name", "parallel", "verbosity"]

        # private
        self._process_name = process_name

    def __repr__(self):
        params = inspect.signature(self.__init__).parameters
        non_default_labels = [label for label in self._labels if getattr(self, label) != params[label].default]
        kws = [f"{key}={getattr(self, key)!r}" for key in non_default_labels]
        return "{}({})".format(type(self).__name__, ", ".join(kws))

    @property
    def process_name(self) -> str:
        """
        The process name given in the :py:meth:`__init__`, if it is ``None`` a default process name is given.
        """
        # set default name
        if self._process_name is None:
            self._process_name = self.process.__str__().replace('\n', '')

        return self._process_name

    def _preprocess(self, ds: DataSet, **kwargs) -> DataSet:
        """
        Applies any data slicing/transformations given in :py:attr:`kwargs` to :py:attr:`ds`.

        Args:
            ds DataSet: The dataset to be pre-processed

            **kwargs: Keyword arguments containing transforms and slicing operations to performed on the data
                before further processing.
        Returns:
            DataSet : Pre-processed data.
        """
        # prep for subclasses _run method

        # grab transformation object
        transform = kwargs.get('transform', lambda x: x)
        ds_new = transform(ds)

        return ds_new

    @abstractmethod
    def _run(self, ds: DataSet, **kwargs) -> dict:
        """
        Method defined for a sub-class. Run the particular process on the data, see for example: :py:class:`Classify`,
        :py:class:`Partition`, and :py:class:`Transform`.

        Args:
            ds (DataSet): The dataset to run the process on.

            **kwargs: Keyword arguments indicating for example the training/test
                labels for that batch, or classification labels for that batch, or a batch-specific transform to apply
                to :py:attr:`ds`.

        Returns:
            dict : Results of the processing, with key labeling the specific results given as values.

        """
        pass

    def run(self, ds: DataSet, batch_args: dict = None) -> Tuple[DataSet, dict]:
        """
        The primary run method. This method calls a sub-classes ``_run`` method on :py:attr:`ds` with keyword arguments
        given by :py:attr:`batch_args[batch]` internally, but takes care of all the
        boiler plate code for running the process across multiple batches in serial or parallel. It collects all of
        the results across batches into a dictionary.

        Args:
            ds (DataSet): The dataset to process.

            batch_args (dict): A dictionary with keys given by batch. Each value in the dictionary is a dictionary of
                keyword arguments to a sub-classes ``_run`` method. A keyword argument may indicate the training/test
                labels for that batch, or classification labels for that batch, or a batch-specific transform to apply
                to :py:attr:`ds`. Note: Batches should be specified by ``batch_0``, ``batch_1``, ... , ``batch_0_0``,
                ``batch_0_0``, etc if you want to link your processes in a :py:attr:`Pipeline` instance,
                In particular ``batch_0_1`` is considered a derivative batch of ``batch_0`` and will inherit if
                possible batch specific transforms, labels, etc... from ``batch_0``.

        Returns:
            Tuple[DataSet, dict] : The first argument is the object :py:attr:`ds` and the second argument is
            :py:attr:`Process.results_`
        """
        if batch_args is None:
            self.results_ = dict(batch=self._run(ds))
            return ds, self.results_

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

    def _run_par(self, ds: DataSet, batch_args: dict) -> dict:
        """
        Runs process on different batches in parallel using `ray <https://ray.io/>`_

        Args:
            ds (DataSet) : See :py:meth:`Process.run`.
            batch_args (dict): See :py:meth:`Process.run`.

        Returns:
            dict : Results across different batches, specifically :py:attr:`Process.results_`.
        """

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

    @staticmethod
    def _find_super_batch(batch: str, kv: dict) -> Union[str, None]:
        """
        Takes the batch label :py:atr:`batch` and attempts to find its super batch in
        :py:attr:`kv`. For example, ``batch_0_1`` would have super batch ``batch_0`` and
        the method would return this if the super batch is in the keys of :py:attr:`kv`.

        Args:
            batch (str): Name of the batch.
            kv (dict): Dictionary to check for super batch in.

        Returns:
            str: Name of the super batch.
        """

        # compute super batch
        super_batch = '_'.join(batch.split('_')[:-1])

        # check if it is in the dictionary
        if super_batch in kv.keys():
            return super_batch
        else:
            return None

    def collapse_results(self, which: Union[list, str] = 'all') -> dict:
        """
        This method collapses the results of the process by batches. Specifically given a key in :py:attr:`which`
        to a result, say ``result_label`` in :py:attr:`self.results_[batch]`, :py:meth:`collapse_results` will call
        a sub-classes :py:meth:`collapse_result_label` method if available, which returns an object containing all
        of the results across batches relevant to ``result_label``. See :py:meth:`Partition._collapse_tvt_labels` for
        an example. In this example, training/test labels can be collapsed into
        a ``DataFrame`` object containing the training/test splits for each batch.

        If a sub-class does not have a method to collapse a specific result across batches, then this method will
        call :py:meth:`Process.extract_result` which returns a dictionary with keys the batches and values the
        batch-specific result.

        This method attempts to collapse the result for all keys listed in :py:attr:`which` and returns a dictionary
        where the keys are :py:attr:`which` and the values are the collapsed results across batches.

        Args:
            which (list or str): List of keys pertaining to the results to be collapsed across batches.

        Returns:
            dict : Contains the collapsed result across batches for each key in :py:attr:`which`.

        Examples:
            >>> # imports
            >>> import os
            >>> from orthrus.core.pipeline import Partition
            >>> from sklearn.model_selection import KFold
            >>> from orthrus.core.dataset import load_dataset
            ...
            >>> # load dataset
            >>> ds = load_dataset(os.path.join(os.environ['ORTHRUS_PATH'],
            ...                                'test_data/Iris/Data/iris.ds'))
            ...
            >>> # define kfold partition
            >>> kfold = Partition(process=KFold(n_splits=5,
            ...                                 shuffle=True,
            ...                                 random_state=124,
            ...                                 ),
            ...                   process_name='5-fold-CV',
            ...                   verbosity=1,
            ...                   )
            ...
            >>> # run process
            >>> ds, results = kfold.run(ds)
            ...
            >>> # print results
            >>> tvt_labels = kfold.collapse_results()['tvt_labels']
            >>> print(tvt_labels)
            ...
            5-fold-CV splits batch_0_split batch_1_split  ... batch_3_split batch_4_split
            0                        Train          Test  ...         Train         Train
            1                         Test         Train  ...         Train         Train
            2                        Train         Train  ...          Test         Train
            3                         Test         Train  ...         Train         Train
            4                        Train         Train  ...         Train         Train
            ..                         ...           ...  ...           ...           ...
            145                      Train          Test  ...         Train         Train
            146                      Train          Test  ...         Train         Train
            147                      Train         Train  ...         Train         Train
            148                      Train         Train  ...          Test         Train
            149                      Train         Train  ...         Train         Train
            [150 rows x 5 columns]
        """
        # check that the process has been run
        assert self.results_ is not None, r"The %s process has not been run yet!" % (self.__class__.__name__,)

        # grab result keys
        keys = list(list(self.results_.values())[0].keys())

        # check if items in 'which' are in the keys
        which = keys if which == 'all' else ([which] if type(which) == str else which)
        assert set(which) <= set(keys), "Some or all of specified results do not exist in the keys!"

        # initialize output
        out = dict()

        # run the associated collapsing methods
        for key in which:
            try:
                result = eval("self._collapse_" + key)()
                out.update({key: result})
            except AttributeError:
                warnings.warn("%s does not contain a collapsing method for %s!"
                              " Returning \"uncollapsed\" object." % (self.__class__.__name__, key))
                result = self.extract_result(key)

            # update dictionary
            out.update({key: result})

        return out

    def extract_result(self, which: str) -> dict:
        """
        For a key, given by :py:attr:`which`, in :py:attr:`Process.results_[batch]` this method creates a dictionary
        with keys batches in :py:attr:`Process.results_` and values :py:attr:`Process.results_[batch][which]`,
        effectively restricting :py:attr:`Process.results_` to only the results related to :py:attr:`which`.

        Args:
            which (str): The key in :py:attr:`Process.results_[batch]` to extract all results across batches with
                respect to.

        Returns:
            dict : Restricted dictionary containing only the results related to :py:attr:`which`.
        """

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
                     collapse: bool = True) -> None:
        """
        Saves the result of the finished process. Objects in collapsed form can either be save as a
        serialized pickle file or a .csv (e.g. numpy.ndarray, pandas.DataFrame, pandas.Series).

        Args:
            save_path (str or dict): If it is a string then the entire dictionary of results is pickled in either
                uncollapsed or collapsed format. If it is a dictionary, the keys should be be the specific results to
                save, with values the individual save paths. Note: :py:attr:`collapse` must be set to ``True`` in order
                to save the individual results.
                
            overwrite (bool): Indicates whether or not to overwrite the existing data specified by :py:attr:`save_path`.

            collapse (bool): Flag indicating whether or not to collapse the results.

        Returns:
            inplace method.
        """

        # make sure there is either a save path or multiple
        assert save_path is not None, "You must provide a save path or multiple save paths in a dictionary!"

        # make sure the type is correct
        assert type(save_path) in [str, dict], "save_path must be a string or a" \
                                               " dictionary with result keys and string values!"

        # acceptable formats
        formats = ['.csv', '.pickle']

        if type(save_path) == str:
            ext = '.' + save_path.split('.')[-1]
            assert ext == '.pickle', "The file extension must be .pickle for saving the results as a whole!"
            if collapse:
                save_object(self.collapse_results(), save_path, overwrite)
            else:
                save_object(self.results_, save_path, overwrite)

        else:
            assert collapse, "Cannot save individual results without collapsing the results first, set collapse=True!"

            # collapse the specificed results
            results = self.collapse_results(which=list(save_path.keys()))

            # loop through each result and save
            for key in save_path.keys():
                # grab the specific result
                result = results[key]

                # check save path
                ind_save_path = generate_save_path(save_path[key], overwrite)

                # grab extension
                ext = '.' + ind_save_path.split('.')[-1]
                not_implemented_for_ext = False

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
                    raise AttributeError("The collapsed result %s may not be saved as %s!" % (key, ext))

        return

    def save(self,
             save_path: str,
             overwrite: bool = False) -> None:

        """
        Save the :py:class:`Process` instance in serialized format using pickle or dill. Calls
        :py:func:`orthrus.core.helper.save_object` internally.

        Args:
            save_path (str): File path to save instance.

            overwrite (bool): If ``True`` :py:attr:`save_path` will be overwritten, if ``False``
                :py:attr:`save_path` will be appended with a version number ``i``, see
                :py:func:`orthrus.core.helper.generate_save_path`.

        Returns:
            File path of saved process.
        """

        # generate new save path in case you don't want to overwrite
        #save_path = generate_save_path(save_path, overwrite)

        # save instance of object
        save_path = save_object(self, save_path, overwrite)

        return save_path


class Partition(Process):
    """
    :py:class:`Process` subclass used to partition a dataset into training, validation, and test samples.

    Parameters:
        process (object): Object to partition the data with, see for example scikit learn's
            `KFold <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html>`_.

        process_name (str): The common name assigned to the :py:attr:`process`.

        parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
            Default is False. :py:func:`ray.init` must be called to initiate the ray cluster before any running
            can be done.

        verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
            the verbosity the larger the text output. Default is 1, indicating the standard text output with a
            :py:class:`Process` instance.

        split_attr (str): Attribute in the dataset's metadata to split with respect to. Default is None.

        split_group (str): Attribute in the dataset's metadata to group with respect to, see
            for example scikit learn's
            `StratifiedShuffleSplit <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html>`_.
            Should be provided when your split need to respect the proportions of the :py:attr:`split_group` class.
            Default is None.

        split_handle (string): Name of ``split`` method used by ``partitioner``. Default is "split".

        split_args (dict): Keyword arguments passed to :py:meth:`process.split()`.

    Attributes:
        process (object): Object to partition the data with, see for example scikit learn's
            `KFold <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html>`_.

        parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
            Default is False:

        verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
            the verbosity the larger the text output. Default is 1, indicating the standard text output with a
            :py:class:`Process` instance.

                split_attr (str): Attribute in the dataset's metadata to split with respect to. Default is None.

        split_group (str): Attribute in the dataset's metadata to group with respect to, see
            for example scikit learn's
            `StratifiedShuffleSplit <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html>`_.
            Should be provided when your split need to respect the proportions of the :py:attr:`split_group` class.
            Default is None.

        split_handle (string): Name of ``split`` method used by ``partitioner``. Default is "split".

        split_args (dict): Keyword arguments passed to :py:meth:`process.split()`.

        run_status_ (int): Indicates whether or not the process has finished. A value of 0 indicates the process has not
            finished, a value of 1 indicated the process has finished.

        results_ (dict of dicts): The results of the run process. The keys of the dictionary indicates the batch results
            for a given batch. For each batch there is a dictionary of results with keys indicating the result type, e.g,
            training/validation/test labels (``tvt_labels``), classification labels (``class_labels``), etc... A
            :py:class:`Partition` instance, after it runs, outputs the following results per batch:

            * tvt_labels (Series): A sample in the series will be labeled either `Train`, `Valid`, or `Test`. If a batch
              already contains training and test labels, then the training samples will be partitioned into training and
              validation. e.g. if batch_0['tvt_labels'] is has training/test labels then the partition process will split
              the training data into training/validation for new batches batch_0_0, batch_0_1, etc... This allows one to
              easily generate training/validation/test labels for a dataset by calling two partition processes back to back.

    Examples:
        >>> # imports
        >>> import os
        >>> from orthrus.core.pipeline import Partition
        >>> from sklearn.model_selection import KFold
        >>> from orthrus.core.dataset import load_dataset
        ...
        >>> # load dataset
        >>> ds = load_dataset(os.path.join(os.environ['ORTHRUS_PATH'],
        ...                                'test_data/Iris/Data/iris.ds'))
        ...
        >>> # define kfold partition
        >>> kfold = Partition(process=KFold(n_splits=5,
        ...                                 shuffle=True,
        ...                                 random_state=124,
        ...                                 ),
        ...                   process_name='5-fold-CV',
        ...                   verbosity=1,
        ...                   )
        ...
        >>> # run process
        >>> ds, results = kfold.run(ds)
        ...
        >>> # print results
        >>> print(results['batch_0']['tvt_labels'])
        ...
        Generating 5-fold-CV splits...
        0      Train
        1       Test
        2      Train
        3       Test
        4      Train
               ...
        145    Train
        146    Train
        147    Train
        148    Train
        149    Train
        Name: 5-fold-CV_0, Length: 150, dtype: object

        >>> # imports
        >>> from sklearn.model_selection import StratifiedShuffleSplit
        ...
        >>> # load dataset
        >>> ds = load_dataset(os.path.join(os.environ['ORTHRUS_PATH'],
        ...                                'test_data/Iris/Data/iris.ds'))
        ...
        >>> # define 80-20 train/test partition
        >>> shuffle = Partition(process=StratifiedShuffleSplit(n_splits=1,
        ...                                                    random_state=113,
        ...                                                    train_size=.8),
        ...                     process_name='80-20-tr-tst',
        ...                     verbosity=1,
        ...                     split_attr ='species',
        ...                     )
        ...
        >>> # run shuffle->kfold
        >>> ds, results = kfold.run(*shuffle.run(ds))
        ...
        >>> # print results
        >>> print("batch_0_0 tvt_labels:\\n%s\\n" %\\
        ...       (results['batch_0_0']['tvt_labels'],))
        ...
        >>> # print train/valid/test counts
        >>> print("batch_0_0 tvt_labels counts:\\n%s" %\\
        ...       (results['batch_0_0']['tvt_labels'].value_counts(),))
        ---------------------
        batch_0_0 tvt_labels:
        0      Train
        1      Valid
        2       Test
        3      Train
        4      Valid
               ...
        145    Train
        146    Train
        147     Test
        148    Train
        149    Train
        Name: 80-20-tr-tst_0_5-fold-CV_0, Length: 150, dtype: object
        ----------------------------
        batch_0_0 tvt_labels counts:
        Train    96
        Test     30
        Valid    24
        Name: 80-20-tr-tst_0_5-fold-CV_0, dtype: int64
    """

    def __init__(self,
                 process: object,
                 process_name: str = None,
                 parallel: bool = False,
                 verbosity: int = 1,
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

        # private attributes
        self._labels += ["split_group", "split_attr"]

    def _run(self, ds: DataSet, **kwargs) -> dict:

        # run the supers _preprocess method
        ds_new = self._preprocess(ds, **kwargs)

        # check for validation in existing labels
        tvt_labels = kwargs.get('tvt_labels', None)

        if tvt_labels is None:
            tvt_labels = pd.Series(index=ds_new.metadata.index, data=['Train']*ds_new.n_samples)

        if (tvt_labels == 'Valid').any():
            raise ValueError("Labels already contain training, validation, and test!")

        # initialize partition result
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
            print(r"Generating %s splits..." % (self.process_name,))

        parts = split(ds_new.data.loc[train_samples], **label_dict, **self.split_args)

        # record training and test labels
        train_test_labels = pd.DataFrame(index=ds_new.metadata.index, columns=[i for i, (train_idx, test_idx) in enumerate(parts)])
        train_test_labels.index.name = ds_new.data.index.name
        parts = split(ds_new.data.loc[train_samples], **label_dict, **self.split_args)  # iterator destroyed

        for i, (train_idx, test_idx) in enumerate(parts):

            if self.verbosity > 1:
                print("Generating split %d...." % (i,))
                
            # create new column for split
            #col_name = i
            train_test_labels[i] = 'Test'

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

    def run(self, ds: DataSet, batch_args: dict = None, append_labels=True) -> Tuple[DataSet, dict]:
        """
        See :py:meth:`Process.run` docstring.

        Args:
            ds (DataSet): See :py:meth:`Process.run` docstring.

            batch_args (dict): See :py:meth:`Process.run` docstring.

            append_labels (bool): If ``tvt_labels`` exist in :py:attr:`batch_args[batch]` then these labels will be
                appended to :py:attr:`Partition.results_`. Useful in the case of splitting training into
                training/validation and wanting to keep the original train/test labels. The default is True.

        Returns:
            Tuple[DataSet, dict] : See :py:meth:`Process.run` docstring.
        """
        # run the super first
        super(Partition, self).run(ds, batch_args)

        # collect super results
        results = dict()

        # split into batches
        for batch in self.results_:
            labels = self.results_[batch]['tvt_labels']

            labels.columns = ['_'.join([str(batch), str(col)]) for col in labels]
            try:
                orig_name = batch_args[batch]['tvt_labels'].name
            except (TypeError, KeyError):
                orig_name = ''
            labels = labels.to_dict('series')
            labels = {k: dict(tvt_labels=v.rename('_'.join([orig_name,
                                                            self.process_name, str(i)]
                                                           ).lstrip('_'))) for i, (k, v) in enumerate(labels.items())}
            results.update(labels)

        # append original labels
        if append_labels and (batch_args is not None):
            if 'tvt_labels' in batch_args[list(batch_args.keys())[0]].keys():
                results.update({k: dict(tvt_labels=v['tvt_labels']) for (k, v) in batch_args.items()})
            try:
                del results['batch']
            except KeyError:
                pass

        # update results
        self.results_ = results

        # change run status
        self.run_status_ = 1

        return ds, self.results_

    def _collapse_tvt_labels(self) -> DataFrame:
        """
        Collapses ``tvt_labels`` into a dataframe where the columns are given as batches and the index is given as
        the samples in the dataset.

        Returns:
            DataFrame : Collapsed train/valid/test labels across batches

        """
        # collapse the dict of dict of series to dataframe
        results = _collapse_dict_dict_pandas(kv=self.results_,
                                             inner_key="tvt_labels")
        return results


class Fit(Process, ABC):
    """
    Base class used for any sub-class of :py:class:`Process` implementing a ``fit`` method, e.g., :py:class:`Transform`,
    :py:class:`Classify`.

    Parameters:
        process (object): Object to fit on the data with, see for example scikit learn's
            `StandardScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_.

        process_name (str): The common name assigned to the :py:attr:`process`.

        parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
            Default is False. :py:func:`ray.init` must be called to initiate the ray cluster before any running
            can be done.

        verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
            the verbosity the larger the text output. Default is 1, indicating the standard text output with a
            :py:class:`Process` instance.

        supervised_attr (str): Supervision attribute in the dataset's metadata to fit with respect to.

        fit_handle (string): Name of ``fit`` method used by :py:attr:`Fit.process`. Default is "fit".

        fit_args (dict): Keyword arguments passed to :py:meth:`process.fit()`.

        prefit (bool): If ``True`` then the process is assumed to be already fit.

    Attributes:
        process (object): Object to fit on the data with, see for example scikit learn's
            `StandardScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_.

        parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
            Default is False:

        verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
            the verbosity the larger the text output. Default is 1, indicating the standard text output with a
            :py:class:`Process` instance.

        supervised_attr (str): Supervision attribute in the dataset's metadata to fit with respect to.

        _fit_handle (string): Name of ``fit`` method used by :py:attr:`Fit.process`. Default is "fit".

        fit_args (dict): Keyword arguments passed to :py:meth:`process.fit()`.

        prefit (bool): If ``True`` then the process is assumed to be already fit.

        run_status_ (int): Indicates whether or not the process has finished. A value of 0 indicates the process has not
            finished, a value of 1 indicated the process has finished.

        results_ (dict of dicts): The results of the run process. The keys of the dictionary indicates the batch results
            for a given batch. For each batch there is a dictionary of results with keys indicating the result type, e.g,
            training/validation/test labels (``tvt_labels``), classification labels (``class_labels``), etc...
    """
    def __init__(self,
                 process: object,
                 process_name: str = None,
                 parallel: bool = False,
                 verbosity: int = 1,
                 supervised_attr: str = None,
                 fit_handle: str = 'fit',
                 fit_args: dict = {},
                 prefit: bool = False,
                 ):

        super(Fit, self).__init__(process=process,
                                  process_name=process_name,
                                  parallel=parallel,
                                  verbosity=verbosity,
                                  )

        # set parameters
        self.supervised_attr = supervised_attr
        self.fit_args = fit_args
        self.prefit = prefit

        # set private attributes
        self._fit_handle = fit_handle
        self._labels += ["fit_args", "supervised_attr"]

    @staticmethod
    def _extract_training_ids(ds: DataSet, **kwargs) -> Series:
        """
        Private method which extracts training indices from existing training/test labels.

        Args:
            ds (DataSet): The dataset to run the process on.

            **kwargs: Possible keyword arguments:
                * ``tvt_labels``: Existing training/test labels to extract training ids from.

        Returns:
            Series : Boolean series with ``True`` indicating the sample is a training sample.
        """
        # grab training labels
        tvt_labels = kwargs.get('tvt_labels', None)
        if tvt_labels is None:
            training_ids = ds.metadata.index
        else:
            training_ids = (tvt_labels == 'Train')

        return training_ids

    def _fit(self, ds: DataSet, **kwargs) -> object:
        """
        Private method for fitting :py:attr:`Fit.process` to a dataset :py:attr:`ds`.

        Args:
            ds (DataSet): The dataset to run the process on.

            **kwargs: Keyword arguments indicating for example the training/test
                labels for that batch, or classification labels for that batch, or a batch-specific transform to apply
                to :py:attr:`ds`.

        Returns:
            object : The fitted :py:attr:`Fit.process`.

        """

        if self.prefit:
            if self.verbosity > 0:
                print(f"Already fit, no fitting performed.")
            return deepcopy(self.process)

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
    """
    :py:class:`Fit` subclass used to transform a dataset, e.g. normalization, imputation, log transformation, etc...

    Parameters:
        process (object): Object to tranform the data with, see for example this packages implementation of
            multi-dimensional scaling: :py:class:`MDS <orthrus.manifold.mds.MDS>`.

        process_name (str): The common name assigned to the :py:attr:`process`.

        parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
            Default is False. :py:func:`ray.init` must be called to initiate the ray cluster before any running
            can be done.

        verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
            the verbosity the larger the text output. Default is 1, indicating the standard text output with a
            :py:class:`Process` instance.

        supervised_attr (str): Supervision attribute in the dataset's metadata to fit with respect to.

        fit_handle (string): Name of ``fit`` method used by :py:attr:`Transform.process`. Default is "fit".

        fit_args (dict): Keyword arguments passed to :py:meth:`process.fit()`.

        prefit (bool): If ``True`` then the process is assumed to be already fit.

        transform_handle (str): Name of ``transform`` method used by :py:attr:`Transform.process`. Default is "transform".

        transform_args (dict): Keyword arguments passed to :py:meth:`process.transform()`.

        fit_transform_handle (str): Name of ``fit_transform`` method used by :py:attr:`Transform.process`.
            Default is "fit_transform".

        retain_f_ids (bool): Flag indicating whether or not to retain the original feature labels. For example,
            some transforms may just transform each individual feature and we would like to keep the name of that
            feature, e.g. log transformation, other transforms will generate latent features which can not be labeled
            with the orginal feature labels, e.g. PCA, MDS, UMAP, etc... The default is ``False``.

        vardata (DataFrame): Optional replacement variable (feature) metadata in the case that ``retain_f_ids`` is ``False``.

    Attributes:
        process (object): Object to tranform the data with, see for example this packages implementation of
            multi-dimensional scaling: :py:class:`MDS <orthrus.manifold.mds.MDS>`.

        process_name (str): The common name assigned to the :py:attr:`process`.

        parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
            Default is False. :py:func:`ray.init` must be called to initiate the ray cluster before any running
            can be done.

        verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
            the verbosity the larger the text output. Default is 1, indicating the standard text output with a
            :py:class:`Process` instance.

        supervised_attr (str): Supervision attribute in the dataset's metadata to fit with respect to.

        _fit_handle (string): Name of ``fit`` method used by :py:attr:`Transform.process`. Default is "fit".

        fit_args (dict): Keyword arguments passed to :py:meth:`process.fit()`.

        prefit (bool): If ``True`` then the process is assumed to be already fit.

        _transform_handle (str): Name of ``transform`` method used by :py:attr:`Transform.process`. Default is "transform".

        transform_args (dict): Keyword arguments passed to :py:meth:`process.transform()`.

        _fit_transform_handle (str): Name of ``fit_transform`` method used by :py:attr:`Transform.process`.
            Default is "fit_transform".

        retain_f_ids (bool): Flag indicating whether or not to retain the original feature labels. For example,
            some transforms may just transform each individual feature and we would like to keep the name of that
            feature, e.g. log transformation, other transforms will generate latent features which can not be labeled
            with the orginal feature labels, e.g. PCA, MDS, UMAP, etc... The default is ``False``.
        
        vardata (DataFrame): Optional replacement variable (feature) metadata in the case that ``retain_f_ids`` is ``False``.
        
        new_f_ids (list): New list of feature ids to replace to original feature ids. Optional.

        run_status_ (int): Indicates whether or not the process has finished. A value of 0 indicates the process has not
            finished, a value of 1 indicated the process has finished.

        results_ (dict of dicts): The results of the run process. The keys of the dictionary indicates the batch results
            for a given batch. For each batch there is a dictionary of results with keys indicating the result type, e.g,
            training/validation/test labels (``tvt_labels``), classification labels (``class_labels``), etc... A
            :py:class:`Transform` instance, after it runs, outputs the following results per batch:

            * transform (Callable): Bound method calling :py:meth:`Transform.process.transform()`
              which is trained on training data in :py:attr:`results_[batch]['tvt_labels']` if it is given, otherwise it
              is trained on all of the data. The bound method can be used to transform future datasets, see the example
              below.

            * transformer (object): The fit transformer generated from :py:attr:`Transform.process`.

    Examples:
        >>> # imports
        >>> import os
        >>> from orthrus.core.pipeline import Transform
        >>> from orthrus.manifold.mds import MDS
        >>> from orthrus.core.dataset import load_dataset
        ...
        >>> # load dataset
        >>> ds = load_dataset(os.path.join(os.environ['ORTHRUS_PATH'],
        ...                                'test_data/Iris/Data/iris.ds'))
        ...
        >>> # define MDS embedding
        >>> mds = Transform(process=MDS(n_components=3),
        ...                 transform_handle=None,
        ...                 process_name='mds',
        ...                 verbosity=1)
        ...
        >>> # run process
        >>> ds, results = mds.run(ds)
        ...
        >>> # use resulting transform
        >>> transform = results['batch']['transform']
        >>> ds_new = transform(ds)
        ...
        >>> # print results
        >>> print(ds_new.data)
        ---------------------------------
                mds_0     mds_1     mds_2
        0   -2.684206  0.326609  0.021512
        1   -2.715399 -0.169557  0.203523
        2   -2.889819 -0.137346 -0.024710
        3   -2.746437 -0.311124 -0.037674
        4   -2.728593  0.333925 -0.096229
        ..        ...       ...       ...
        145  1.944017  0.187415 -0.179303
        146  1.525663 -0.375021  0.120637
        147  1.764046  0.078520 -0.130784
        148  1.901629  0.115877 -0.722873
        149  1.389666 -0.282887 -0.362318
        _
        [150 rows x 3 columns]
    """

    def __init__(self,
                 process: object,
                 process_name: str = None,
                 parallel: bool = False,
                 verbosity: int = 1,
                 retain_f_ids: bool = False,
                 vardata: DataFrame = None,
                 new_f_ids: list = None,
                 fit_handle: str = 'fit',
                 transform_handle: str = 'transform',
                 fit_transform_handle: str = 'fit_transform',
                 supervised_attr: str = None,
                 fit_args: dict = {},
                 prefit: bool = False,
                 transform_args: dict = {}):

        # init with Process class
        super(Transform, self).__init__(process=process,
                                        process_name=process_name,
                                        parallel=parallel,
                                        verbosity=verbosity,
                                        supervised_attr=supervised_attr,
                                        fit_handle=fit_handle,
                                        fit_args=fit_args,
                                        prefit=prefit,
                                        )

        # set parameters
        self.retain_f_ids = retain_f_ids
        self.vardata = vardata
        self.new_f_ids = new_f_ids
        self.transform_args = transform_args

        # set private attributes
        self._transform_handle = transform_handle
        self._fit_transform_handle = fit_transform_handle
        self._labels += ["retain_f_ids", "transform_args", "vardata"]

        # check appropriate parameters
        if self._fit_handle is None or self._transform_handle is None:
            if self._fit_transform_handle is None:
                raise ValueError("%s process must have either both a fit method and a"
                                 " transform method or just a fit_transform method!" % (self.__class__.__name__,))
            else:
                warnings.warn("%s will use its fit_transform method to fit."
                              " Make sure that its fit_transform method fits"
                              " the transformation inplace!" % (self.__class__.__name__,))

    def _fit_transform(self, ds: DataSet, **kwargs) -> Tuple[object, DataSet]:
        """
        Private method for fit transforming a dataset :py:attr:`ds`. with :py:attr:`Transform.process`.

        Args:
            ds (DataSet): The dataset to run the process on.

            **kwargs: Keyword arguments indicating for example the training/test
                labels for that batch, or classification labels for that batch, or a batch-specific transform to apply
                to :py:attr:`ds`.

        Returns:
            tuple :
                * object: The fitted :py:attr:`Transform.process`.
                * DataSet: The transformed data.
        """

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

    def _run(self, ds: DataSet, **kwargs) -> dict:

        # initalize output
        result = dict()

        # run the super's _preprocess method
        ds_new = self._preprocess(ds, **kwargs)

        # attempt to fit
        if self._fit_handle is not None:
            process = self._fit(ds_new, **kwargs)
        else:
            process, _ = self._fit_transform(ds_new, **kwargs)

        # store the resulting transformer
        result['transformer'] = process

        # generate its transform and store it
        result['transform'] = self._generate_transform(process)

        return result

    def _generate_transform(self, process: object) -> Callable:
        """
        Private method for generating a callable transform function on a dataset.

        Args:
            process (object): The transformer object to extract the transform from.

        Returns:
            Callable : A bound method to transform datasets with. Calls :py:meth:`Transform.process.transform()`
                internally.
        """

        # check for a transform handle
        if self._transform_handle is not None:
            inner_transform = eval("process." + self._transform_handle)

        # define transform
        def transform(ds: DataSet):
            if self._transform_handle is None:
                _, data_new = self._fit_transform(ds)
                data_new = np.array(data_new)
            else:
                if self.verbosity > 0:
                    print(r"Transforming the data using %s..." % (self.process_name,))
                data_new = np.array(inner_transform(ds.data, **self.transform_args))
            if self.retain_f_ids:
                try:
                    data_new = pd.DataFrame(data=data_new, index=ds.data.index, columns=ds.data.columns)
                    ds_new = deepcopy(ds)
                    ds_new.data = data_new
                except ValueError:
                    raise ValueError("Transform changes the dimension of the data and therefore cannot retain"
                                     " the original feature ids in the new dataset!")
            else:
                if self.new_f_ids is None:
                    columns = ['_'.join([self.process_name, str(i)]) for i in range(data_new.shape[1])]
                else:
                    columns = self.new_f_ids
                    
                data_new = pd.DataFrame(data=data_new, index=ds.data.index,
                                        columns=columns)
                # check if features are the same after transformation and use 6original feature ids for columns
                ds_new = DataSet(data=data_new, metadata=ds.metadata, vardata=self.vardata)

            return ds_new

        return transform

    def transform(self, ds: DataSet) -> dict:
        """
        Transforms the dataset :py:attr:`ds` for every transform contained within :py:attr:`Transform.results_`.

        Args:
            ds (DataSet): The dataset to transform.

        Returns:
            dict : The keys indicate the batch in :py:attr:`Transform.results_` and the values are the transformed
            datasets given by :py:attr:`Transform.results_[batch]['transform'](ds)`.
        """
        assert self.results_ is not None, "Transform must call its run method on a dataset" \
                                          " before it can transform a new dataset!"

        # transform the incoming data according to transforms
        out = {k: v['transform'](ds) for (k, v) in self.results_.items()}
        return out


class FeatureSelect(Transform):
    """
    :py:class:`Transform` subclass used to select and restrict features in a dataset.

    Parameters:
        process (object): Object to feature select and restrict the data with, see for example this packages
            implementation of k-fold feature selection: :py:class:`KFFS <orthrus.sparse.feature_selection.kffs.KFFS>`.

        process_name (str): The common name assigned to the :py:attr:`process`.

        parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
            Default is False. :py:func:`ray.init` must be called to initiate the ray cluster before any running
            can be done.

        verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
            the verbosity the larger the text output. Default is 1, indicating the standard text output with a
            :py:class:`Process` instance.

        supervised_attr (str): Supervision attribute in the dataset's metadata to fit with respect to.

        fit_handle (string): Name of ``fit`` method used by :py:attr:`FeatureSelect.process`. Default is "fit".

        fit_args (dict): Keyword arguments passed to :py:meth:`process.fit()`.

        prefit (bool): If ``True`` then the process is assumed to be already fit.

        transform_handle (str): Name of ``transform`` method used by :py:attr:`FeatureSelect.process`.
            Default is "transform".

        transform_args (dict): Keyword arguments passed to :py:meth:`process.transform()`.

        fit_transform_handle (str): Name of ``fit_transform`` method used by :py:attr:`FeatureSelect.process`.
            Default is "fit_transform".

        f_ranks_handle (str): Name of the attribute in :py:attr:`FeatureSelect.process` containing the feature ranks.
            Default is None.

    Attributes:
        process (object): Object to feature select and restrict the data with, see for example this packages
            implementation of k-fold feature selection: :py:class:`KFFS <orthrus.sparse.feature_selection.kffs.KFFS>`.

        process_name (str): The common name assigned to the :py:attr:`process`.

        parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
            Default is False. :py:func:`ray.init` must be called to initiate the ray cluster before any running
            can be done.

        verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
            the verbosity the larger the text output. Default is 1, indicating the standard text output with a
            :py:class:`Process` instance.

        supervised_attr (str): Supervision attribute in the dataset's metadata to fit with respect to.

        _fit_handle (string): Name of ``fit`` method used by :py:attr:`FeatureSelect.process`. Default is "fit".

        fit_args (dict): Keyword arguments passed to :py:meth:`process.fit()`.

        prefit (bool): If ``True`` then the process is assumed to be already fit.

        _transform_handle (str): Name of ``transform`` method used by :py:attr:`FeatureSelect.process`.
            Default is "transform".

        transform_args (dict): Keyword arguments passed to :py:meth:`process.transform()`.

        _fit_transform_handle (str): Name of ``fit_transform`` method used by :py:attr:`FeatureSelect.process`.
            Default is "fit_transform".

        _f_ranks_handle (str): Name of the attribute in :py:attr:`FeatureSelect.process` containing the feature ranks.
            Default is None.

        run_status_ (int): Indicates whether or not the process has finished. A value of 0 indicates the process has not
            finished, a value of 1 indicated the process has finished.

        results_ (dict of dicts): The results of the run process. The keys of the dictionary indicates the batch results
            for a given batch. For each batch there is a dictionary of results with keys indicating the result type, e.g,
            training/validation/test labels (``tvt_labels``), classification labels (``class_labels``), etc... A
            :py:class:`FeatureSelect` instance, after it runs, outputs the following results per batch:

            * transform (Callable): Bound method calling :py:meth:`FeatureSelect.process.transform()`
              which is trained on training data in :py:attr:`results_[batch]['tvt_labels']` if it is given, otherwise it
              is trained on all of the data. The bound method can be used to restrict future datasets to the
              selected features, see the example below.

            * selector (object): The fit feature selector generated from :py:attr:`FeatureSelect.process`.

            * f_ranks (Series): Feature ranks given by the feature selector. The index of the ``Series``
              is given by the features in the dataset and the values are the feature ranks determined by the
              feature selector.

    Examples:
        >>> # imports
        >>> import os
        >>> from orthrus.core.pipeline import FeatureSelect
        >>> from orthrus.sparse.feature_selection.kffs import KFFS
        >>> from sklearn.svm import LinearSVC
        >>> import numpy as np
        >>> from orthrus.core.dataset import load_dataset
        ...
        >>> # load dataset
        >>> ds = load_dataset(os.path.join(os.environ['ORTHRUS_PATH'],
        ...                                'test_data/GSE73072/Data/GSE73072.ds'))
        ...
        >>> # speed up things for this example
        >>> ds = ds.slice_dataset(feature_ids=ds.vardata.index[:1000])
        ...
        >>> # define KFFS feature selector
        >>> kffs = FeatureSelect(process=KFFS(classifier=LinearSVC(penalty='l1',
        ...                                                        dual=False),
        ...                                   f_weights_handle='coef_',
        ...                                   f_rnk_func=np.abs,
        ...                                   random_state=235,
        ...                                   ),
        ...                      process_name='kffs',
        ...                      supervised_attr='Shedding',
        ...                      transform_args=dict(n_top_features=100),
        ...                      f_ranks_handle='ranks_',
        ...                      verbosity=1)
        ...
        >>> # run process
        >>> ds, results = kffs.run(ds)
        ...
        >>> # use resulting transform
        >>> transform = results['batch']['transform']
        >>> ds_new = transform(ds)
        ...
        >>> # print results
        >>> print(ds_new.data)
        ---------------------------------
        ID_REF       1773_at  200056_s_at  ...  201427_s_at  201462_at
        GSM1881744  6.964445     7.486071  ...     3.658097   8.622978
        GSM1881745  7.162511     7.434805  ...     3.580072   8.667888
        GSM1881746  7.071087     7.809637  ...     3.596919   8.432335
        GSM1881747  6.943840     7.549568  ...     3.572631   8.585819
        GSM1881748  6.937150     7.687864  ...     3.893286   8.625159
        ...              ...          ...  ...          ...        ...
        GSM1884625  6.748496     6.635707  ...     3.699136   8.147086
        GSM1884626  6.467847     6.055161  ...     3.609685   7.171061
        GSM1884627  6.474651     7.860354  ...     3.959776   7.848822
        GSM1884628  7.078167     7.508468  ...     3.583747   8.158019
        GSM1884629  6.457082     7.465501  ...     3.953059   7.773569
        _
        [2886 rows x 100 columns]
    """
    def __init__(self,
                 process: object,
                 process_name: str = None,
                 parallel: bool = False,
                 verbosity: int = 1,
                 fit_handle: str = 'fit',
                 transform_handle: str = 'transform',
                 supervised_attr: str = None,
                 fit_args: dict = {},
                 prefit: bool = False,
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
                                            prefit=prefit,
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
        self._labels.remove("retain_f_ids")

    def _preprocess(self, ds: DataSet, **kwargs) -> DataSet:
        return ds  # avoid double preprocessing from Transform

    def _run(self, ds: DataSet, **kwargs) -> dict:

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

    def _generate_transform(self, process: object = None) -> Callable:
        # generate selector
        select = eval("process." + self._transform_handle)

        # define transform
        def transform(ds: DataSet):
            if self.verbosity > 0:
                print(r"Restricting features in the data using %s..." % (self.process_name,))
            feature_ids = np.array(select(ds.data.columns.to_numpy().reshape(1, -1), **self.transform_args)).reshape(-1,).tolist()
            ds_new = ds.slice_dataset(feature_ids=feature_ids)

            return ds_new

        return transform

    def _generate_f_ranks(self, process: object, ds: DataSet) -> dict:
        """
        Private method for generating feature ranks from :py:attr:`FeatureSelect.process` using
        :py:attr:`FeatureSelect._f_ranks_handle`.

        Args:
            process (object): Object to extract feature ranks from.
            ds (DataSet): Dataset used to extract features.

        Returns:
            dict : key="f_ranks", value=Series of feature ranks indexed by
            :py:attr:`vardata.index <orthrus.core.dataset.DataSet.vardata>`
        """
        # initialize out
        out = {}

        # check for f_ranks attribute
        if self._f_ranks_handle is not None:
            f_ranks = eval(f"process.{self._f_ranks_handle}")
            if type(f_ranks) == DataFrame:
                f_ranks.rename(index=dict(zip(f_ranks.index.tolist(), ds.vardata.index.tolist())), inplace=True)
                #f_ranks.columns.name = self.process_name + " f_ranks"
            else:
                f_ranks = np.array(f_ranks)  # convert to ndarray

                # check shape
                if len(f_ranks.shape) == 1:
                    f_ranks = Series(index=ds.vardata.index,
                                     data=f_ranks,
                                     )
                else:
                    f_ranks = DataFrame(index=ds.vardata.index,
                                        data=f_ranks)
                    f_ranks.index.name = ds.vardata.index.name

            # update output
            out.update({'f_ranks': f_ranks})

        return out

    def _collapse_f_ranks(self) -> dict:
        """
        Collapses ``f_ranks`` into a dataframe where the columns are given as batches and the index is given as
        the features in the dataset.

        Returns:
            DataFrame : Collapsed feature ranks across batches

        """
        # collapse the dict of dict of dataframe to dataframe
        results = _collapse_dict_dict_pandas(kv=self.results_,
                                             inner_key="f_ranks",
                                             )
        return results

    def transform(self, ds: DataSet) -> dict:
        """
        Transforms the dataset :py:attr:`ds` for every transform contained within :py:attr:`FeatureSelect.results_`.

        Args:
            ds (DataSet): The dataset to restrict features with.

        Returns:
            dict : The keys indicate the batch in :py:attr:`FeatureSelect.results_` and the values are the restricted
            datasets given by :py:attr:`FeatureSelect.results_[batch]['transform'](ds)`.
        """
        assert self.results_ is not None, "Transform must call its run method on a dataset" \
                                          " before it can transform a new dataset!"

        # transform the incoming data according to transforms
        out = {k: v['transform'](ds) for (k, v) in self.results_.items()}
        return out


class Classify(Fit):
    """
        :py:class:`Fit` subclass used to classify a dataset.

        Parameters:
            process (object): Object to classify the data with, see for example scikit-learn's
                `LinearSVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html>`_.

            process_name (str): The common name assigned to the :py:attr:`process`.

            parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
                Default is False. :py:func:`ray.init` must be called to initiate the ray cluster before any running
                can be done.

            verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
                the verbosity the larger the text output. Default is 1, indicating the standard text output with a
                :py:class:`Process` instance.

            class_attr (str): Attribute in the dataset's metadata to classify with respect to.

            fit_handle (string): Name of ``fit`` method used by :py:attr:`Classify.process`. Default is "fit".

            fit_args (dict): Keyword arguments passed to :py:meth:`process.fit()`.

            predict_handle (str): Name of ``predict`` method used by :py:attr:`Classify.process`. Default is "predict".

            predict_args (dict): Keyword arguments passed to :py:meth:`process.predict()`.

            classes_handle (str): Name of attribute in :py:attr:`Classify.process` contain the list of class labels.
                The default is scikit-learn's default "classes_".

            f_weights_handle (string): Name of :py:attr:`Classify.process` attribute containing feature weights.
                Default is None.

            s_weights_handle (string): Name of :py:attr:`Classify.process` attribute containing sample weights.
                Default is None.

        Attributes:
            process (object): Object to classify the data with, see for example scikit-learn's
                `LinearSVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html>`_.

            process_name (str): The common name assigned to the :py:attr:`process`.

            parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
                Default is False. :py:func:`ray.init` must be called to initiate the ray cluster before any running
                can be done.

            verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
                the verbosity the larger the text output. Default is 1, indicating the standard text output with a
                :py:class:`Process` instance.

            class_attr (str): Attribute in the dataset's metadata to classify with respect to.

            _fit_handle (string): Name of ``fit`` method used by :py:attr:`Classify.process`. Default is "fit".

            fit_args (dict): Keyword arguments passed to :py:meth:`process.fit()`.

            _predict_handle (str): Name of ``predict`` method used by :py:attr:`Classify.process`. Default is "predict".

            predict_args (dict): Keyword arguments passed to :py:meth:`process.predict()`.

            _classes_handle (str): Name of attribute in :py:attr:`Classify.process` contain the list of class labels.
                The default is scikit-learn's default "classes_".

            _f_weights_handle (string): Name of :py:attr:`Classify.process` attribute containing feature weights.
                Default is None.

            _s_weights_handle (string): Name of :py:attr:`Classify.process` attribute containing sample weights.
                Default is None.

            run_status_ (int): Indicates whether or not the process has finished. A value of 0 indicates the process has not
                finished, a value of 1 indicated the process has finished.

            results_ (dict of dicts): The results of the run process. The keys of the dictionary indicates the batch results
                for a given batch. For each batch there is a dictionary of results with keys indicating the result type, e.g,
                training/validation/test labels (``tvt_labels``), classification labels (``class_labels``), etc... A
                :py:class:`Classify` instance, after it runs, outputs the following results per batch:

                * class_labels (Series): Prediction labels generated by the classifier, the index of the ``Series``
                  is given by the samples in the dataset. The values of the ``Series`` will be labels contained in
                  :py:attr:`Classify.process.classes_`. The classifier is fit only on the training data in
                  :py:attr:`results_[batch]['tvt_labels']` if it is given, otherwise it is trained on all of the data.

                * class_scores (Series or DataFrame): Prediction scores generated by the classifier,
                  the index of the ``Series`` or ``DataFrame`` is given by the samples in the dataset.
                  The columns in the ``DataFrame`` are given by the classes in :py:attr:`Classify.process.classes_`, the
                  values of the ``DataFrame`` will be scores indicating the strength of membership to a specific class.
                  The classifier is fit only on the training data in :py:attr:`results_[batch]['tvt_labels']` if it
                  is given, otherwise it is trained on all of the data.

                * classifier (object): The fit classifier generated from :py:attr:`Classify.process`

                * f_weights (Series): Feature weights or importances given by the classifier. The index of the ``Series``
                  is given by the features in the dataset and the values are the feature weights determined by the
                  classifier.

                * s_weights (Series): Sample weights or importances given by the classifier. The index of the ``Series``
                  is given by the samples in the dataset and the values are the sample weights determined by the
                  classifier.

        Examples:
            >>> # imports
            >>> import os
            >>> from orthrus.core.pipeline import Classify, Partition
            >>> from sklearn.ensemble import RandomForestClassifier as RFC
            >>> from sklearn.model_selection import StratifiedShuffleSplit
            >>> from orthrus.core.dataset import load_dataset
            ...
            >>> # load dataset
            >>> ds = load_dataset(os.path.join(os.environ['ORTHRUS_PATH'],
            ...                                'test_data/Iris/Data/iris.ds'))
            ...
            >>> # define 80-20 train/test partition
            >>> shuffle = Partition(process=StratifiedShuffleSplit(n_splits=1,
            ...                                                    random_state=113,
            ...                                                    train_size=.8),
            ...                     process_name='80-20-tr-tst',
            ...                     verbosity=1,
            ...                     split_attr ='species',
            ...                     )
            ...
            >>> # define random forest classify process
            >>> rf = Classify(process=RFC(),
            ...                process_name='RF',
            ...                class_attr='species',
            ...                verbosity=1)
            ...
            >>> # run process
            >>> ds, results = rf.run(*shuffle.run(ds))
            ...
            >>> # print results
            >>> print(results['batch_0']['class_labels'])
            ---------------------------------
            0         setosa
            1         setosa
            2         setosa
            3         setosa
            4         setosa
                     ...
            145    virginica
            146    virginica
            147    virginica
            148    virginica
            149    virginica
            Name: RF labels, Length: 150, dtype: object

            >>> # define random forest classify process using probabilities
            >>> rf = Classify(process=RFC(),
            ...                process_name='RF',
            ...                class_attr='species',
            ...                predict_handle='predict_proba',
            ...                verbosity=1)
            ...
            >>> # run process
            >>> ds, results = rf.run(*shuffle.run(ds))
            ...
            >>> # print results
            >>> print(results['batch_0']['class_scores'])
            ---------------------------------
            RF scores  setosa  versicolor  virginica
            0             1.0        0.00       0.00
            1             1.0        0.00       0.00
            2             1.0        0.00       0.00
            3             1.0        0.00       0.00
            4             1.0        0.00       0.00
            ..            ...         ...        ...
            145           0.0        0.01       0.99
            146           0.0        0.00       1.00
            147           0.0        0.00       1.00
            148           0.0        0.00       1.00
            149           0.0        0.03       0.97
            _
            [150 rows x 3 columns]
        """
    def __init__(self,
                 process: object,
                 class_attr: str,
                 process_name: str = None,
                 parallel: bool = False,
                 verbosity: int = 1,
                 fit_handle: str = 'fit',
                 predict_handle: str = 'predict',
                 fit_args: dict = {},
                 predict_args: dict = {},
                 classes_handle: str = 'classes_',
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
        self.predict_args = predict_args
        self.class_attr = self.supervised_attr  # shadows supervised_attr

        # set private attributes
        self._predict_handle = predict_handle
        self._classes_handle = classes_handle
        self._f_weights_handle = f_weights_handle
        self._s_weights_handle = s_weights_handle
        self._labels.remove("supervised_attr")
        self._labels += ["class_attr", "predict_args"]

        # check appropriate parameters
        if self._fit_handle is None or self._predict_handle is None:
            raise ValueError("Classify process must have both a fit method and a predict method!")

    def _collapse_class_labels(self) -> DataFrame:
        """
        Collapses ``class_labels`` into a dataframe where the columns are given as batches and the index is given as
        the samples in the dataset.

        Returns:
            DataFrame : Collapsed prediction labels across batches

        """

        # collapse the dict of dict of series to dataframe
        results = _collapse_dict_dict_pandas(kv=self.results_,
                                             inner_key="class_labels",
                                             columns_name=self.process_name + " labels",
                                             col_suffix='_'.join([self.process_name, "labels"]))
        return results

    def _collapse_class_scores(self) -> DataFrame:
        """
        Collapses ``class_scores`` into a dataframe where the columns are given as ``batch``_``class`` for ``batch`` in
        :py:attr:`Classify.results_` and ``class`` in :py:attr:`Classify.process.classes_`. The index is given as
        the samples in the dataset.

        Returns:
            DataFrame : Collapsed prediction scores across batches

        """
        # collapse the dict of dict of dataframe to dataframe
        results = _collapse_dict_dict_pandas(kv=self.results_,
                                             inner_key="class_scores",
                                             columns_name=self.process_name + " scores",
                                             col_suffix="scores")
        return results

    def _collapse_f_weights(self) -> DataFrame:
        """
        Collapses ``f_weights`` into a dataframe where the columns are given as batches and the index is given as
        the features in the dataset.

        Returns:
            DataFrame : Collapsed feature weights across batches

        """
        # collapse the dict of dict of series to dataframe
        results = _collapse_dict_dict_pandas(kv=self.results_,
                                             inner_key="f_weights",
                                             columns_name=self.process_name + " f_weights",
                                             col_suffix="f_weights")

        return results

    def _collapse_s_weights(self) -> DataFrame:
        """
        Collapses ``s_weights`` into a dataframe where the columns are given as batches and the index is given as
        the samples in the dataset.

        Returns:
            DataFrame : Collapsed sample weights across batches

        """
        # collapse the dict of dict of series to dataframe
        results = _collapse_dict_dict_pandas(kv=self.results_,
                                             inner_key="s_weights",
                                             columns_name=self.process_name + " s_weights",
                                             col_suffix="s_weights")

        return results

    def _run(self, ds: DataSet, **kwargs) -> dict:

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
        """
        Private method used to create classification labels or scores for :py:attr:`ds`
        using :py:attr:`Classify.process`.

        Args:
            process (object): Classifier used to make predictions.
            ds (DataSet): Dataset to be used to make predictions on.

        Returns:
            dict : key = "class_labels" or "class_scores", value = ``Series`` or ``DataFrame`` with classification labels
            or scores.

        """

        # predict on dataset
        if self.verbosity > 0:
            print(r"Classifying the data using %s..." % (self.process_name,))
        predictions = eval("process." + self._predict_handle)(ds.data)

        # format output as series if labels
        if len(predictions.shape) == 2:
            pred = pd.DataFrame(data=predictions,
                                index=ds.metadata.index,
                                columns=eval("process." + self._classes_handle))
            pred.columns.name = self.process_name + " scores"
            pred_label = 'class_scores'
        else:
            pred = pd.Series(name=self.process_name + " labels", data=predictions, index=ds.metadata.index)
            pred_label = 'class_labels'

        return {pred_label: pred}

    def _generate_f_s_weights(self, process: object, ds: DataSet) -> dict:
        """
        Private method used to generate feature or sample weights for :py:attr:`ds`
        using :py:attr:`Classify.process`.

        Args:
            process (object): Classifier used to make predictions.
            ds (DataSet): Dataset to be used to make predictions on.

        Returns:
            dict : keys = {"f_weights", "s_weights", values = {``Series`` with feature weights, ``Series`` with sample
            weights}.

        """

        # initialize output
        out = dict()

        # extract feature weights
        if self._f_weights_handle is not None:
            f_weights = pd.Series(index=ds.vardata.index,
                                  data=np.array(eval("process." + self._f_weights_handle)).reshape(-1,),
                                  name='_'.join([self.process_name, 'f_weights']))

            out.update({'f_weights': f_weights})

        # extract sample weights
        if self._s_weights_handle is not None:
            s_weights = pd.Series(index=ds.metadata.index,
                                  data=eval("process." + self._s_weights_handle),
                                  name='_'.join([self.process_name, 's_weights']))

            out.update({'s_weights': s_weights})

        return out


class Score(Process):
    """
    :py:class:`Process` subclass used to score classification and regression results.

    Parameters:
        process (Callable): The function used to score the classification results.

        process_name (str): The common name assigned to the :py:attr:`process`.

        parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
            Default is False. :py:func:`ray.init` must be called to initiate the ray cluster before any running
            can be done.

        verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
            the verbosity the larger the text output. Default is 1, indicating the standard text output with a
            :py:class:`Process` instance.

        score_args (dict): Keyword arguments passed to :py:attr:`Score.process()`.

        pred_type (str): Can be either "class_labels", "class_scores", "reg_scores", currently. It indicates the
            type predictions made, i.e., classification labels, classification scores, or regression scores. The
            default is "class_labels".

        sample_weight_attr (str): Attribute in the metadata of the dataset you wish to weight the scores by, e.g.,
            misclassifying a sick sample might be more costly than misclassifying a healthy sample.

        infer_class_labels_on_output (bool): If ``True`` the process will attempt to assign labels to the output score.
            For example if one uses a confusion matrix the process will attempt to assign the class labels given in
            :py:attr:`Score.classes` to the rows and columns for indexing.

        classes (list): Classes used for classification labels. You can provide a subset of classification labels
            to look at scores relative to fewer classes. The default is None.

    Attributes:
        process (Callable): The function used to score the classification results.

        process_name (str): The common name assigned to the :py:attr:`process`.

        parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
            Default is False. :py:func:`ray.init` must be called to initiate the ray cluster before any running
            can be done.

        verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
            the verbosity the larger the text output. Default is 1, indicating the standard text output with a
            :py:class:`Process` instance.

        score_args (dict): Keyword arguments passed to :py:attr:`Score.process()`.

        pred_type (str): Can be either "class_labels", "class_scores", "reg_scores", currently. It indicates the
            type predictions made, i.e., classification labels, classification scores, or regression scores. The
            default is "class_labels".

        _sample_weight_attr (str): Attribute in the metadata of the dataset you wish to weight the scores by, e.g.,
            misclassifying a sick sample might be more costly than misclassifying a healthy sample.

        _infer_class_labels_on_output (bool): If ``True`` the process will attempt to assign labels to the output score.
            For example if one uses a confusion matrix the process will attempt to assign the class labels given in
            :py:attr:`Score.classes` to the rows and columns for indexing.

        _classes (list): Classes used for classification labels. You can provide a subset of classification labels
            to look at scores relative to fewer classes. The default is None.

        run_status_ (int): Indicates whether or not the process has finished. A value of 0 indicates the process has not
            finished, a value of 1 indicated the process has finished.

        results_ (dict of dicts): The results of the run process. The keys of the dictionary indicates the batch results
            for a given batch. For each batch there is a dictionary of results with keys indicating the result type, e.g,
            training/validation/test labels (``tvt_labels``), classification labels (``class_labels``), etc...
            A :py:class:`Score` instance, after it runs, outputs the following results per batch:

            * class_pred_scores (Series): Scores generated by :py:attr:`Score.process` on classification results
              generated by :py:class:`Classify`. The index of the ``Series`` is given by the labels in
              ``batch['tvt_labels']``, e.g., "Train", "Valid", "Test". The values of the ``Series`` are the associated
              scores for each sample type: "Train", "Valid", "Test".

            * reg_pred_scores (Series): Scores generated by :py:attr:`Score.process` on regression results
              generated by :py:class:`Regress`. The index of the ``Series`` is given by the labels in
              ``batch['tvt_labels']``, e.g., "Train", "Valid", "Test". The values of the ``Series`` are the associated
              scores for each sample type: "Train", "Valid", "Test".

    Examples:
            >>> # imports
            >>> import os
            >>> from orthrus.core.pipeline import Score, Classify, Partition
            >>> from sklearn.ensemble import RandomForestClassifier as RFC
            >>> from sklearn.model_selection import StratifiedShuffleSplit
            >>> from sklearn.metrics import balanced_accuracy_score
            >>> from orthrus.core.dataset import load_dataset
            ...
            >>> # load dataset
            >>> ds = load_dataset(os.path.join(os.environ['ORTHRUS_PATH'],
            ...                                'test_data/Iris/Data/iris.ds'))
            ...
            >>> # define 80-20 train/test partition
            >>> shuffle = Partition(process=StratifiedShuffleSplit(n_splits=1,
            ...                                                    random_state=113,
            ...                                                    train_size=.8),
            ...                     process_name='80-20-tr-tst',
            ...                     verbosity=1,
            ...                     split_attr ='species',
            ...                     )
            ...
            >>> # define random forest classify process
            >>> rf = Classify(process=RFC(),
            ...                process_name='RF',
            ...                class_attr='species',
            ...                verbosity=1)
            ...
            >>> # define balance accuracy score process
            >>> bsr = Score(process=balanced_accuracy_score,
            ...             process_name='bsr',
            ...             pred_attr='species',
            ...             verbosity=2)
            ...
            >>> # run partition and classification processes
            >>> ds, results_0 = shuffle.run(ds)
            >>> ds, results_1 = rf.run(ds, results_0)
            ...
            >>> # carry over tvt_labels, use Pipeline for chaining processes instead!
            >>> [results_1[batch].update(results_0[batch]) for batch in results_1]
            ...
            >>> # score classification results
            >>> ds, results = bsr.run(ds, results_1)
            -----------
            bsr scores:
            Train: 100.00%
            Test: 96.67%
    """

    def __init__(self,
                 process: Callable,
                 pred_attr: Union[str, list],
                 process_name: str = None,
                 parallel: bool = False,
                 verbosity: int = 1,
                 score_args: dict = {},
                 pred_type: str = 'class_labels',
                 sample_weight_attr: str = None,
                 infer_class_labels_on_output: bool = True,
                 classes: list = None,
                 ):

        # init with Process class
        super(Score, self).__init__(process=process,
                                    process_name=process_name,
                                    parallel=parallel,
                                    verbosity=verbosity,
                                    )
        # parameters
        self.pred_type = pred_type
        self.pred_attr = pred_attr
        self.score_args = score_args

        # private attributes
        self._classes = classes
        self._sample_weight_attr = sample_weight_attr
        self._infer_class_labels_on_output = infer_class_labels_on_output
        self._labels += ["pred_type", "pred_attr", "score_args"]

    def _run(self, ds: DataSet, **kwargs) -> dict:

        # preprocess data
        ds_new = self._preprocess(ds, **kwargs)

        # grab tvt labels
        tvt_labels = kwargs.get('tvt_labels', pd.Series(index=ds_new.metadata.index, data=['Train']*ds_new.n_samples))
        score_rows = tvt_labels.unique()

        # grab classification labels/scores
        pred_type = self.pred_type
        y_pred = kwargs.get(pred_type, None)
        assert y_pred is not None, "%s process requires predictions in" \
                                   " order to compute metrics!" % (self.__class__.__name__,)

        # initialize result
        result = dict()

        # decorate scoring object based on task
        if pred_type == "class_labels":
            scorer = self._process_classification_labels(self.process)
        elif pred_type == "class_scores":
            scorer = self._process_classification_scores(self.process)
        elif pred_type == "reg_values":
            scorer = self._process_regression_values(self.process)
        else:
            raise NotImplementedError("%s currently can not handle scoring %s,"
                                      " check your pred_type attribute!" % (self.__class__.__name__, self.pred_type))

        # initialize output
        scores = pd.Series(index=score_rows, name=self.process_name)

        # grab true labels
        if type(self.pred_attr) == str:
            pred_attr = [self.pred_attr]
        else:
            pred_attr = self.pred_attr
        y_true = pd.DataFrame(index=ds_new.metadata.index, data=ds_new.metadata[pred_attr].values, columns=pred_attr)

        # restrict class labels/scores and tvt_labels to y_true
        y_pred = deepcopy(y_pred.loc[y_true.index])
        tvt_labels = deepcopy(tvt_labels.loc[y_true.index])

        # make a deep copy of the score_args (dictionaries are mutable!)
        score_args = deepcopy(self.score_args)

        # check for labels
        if 'class' in pred_type:
            # get labels arg from scorer arguments
            labels = score_args.get('labels', None)

            # if _classes is provided then use this for sort order and restricting
            if self._classes is not None:
                score_args['labels'] = self._classes
                if labels is not None:
                    # Let the user know we are ignoring their input here
                    warnings.warn("Both %s._classes and labels in score_args is given!"
                                  " %s will use %s._classes inplace of labels for sorting"
                                  " and restricting score output." % (self.__class__.__name__,
                                                                      self.__class__.__name__,
                                                                      self.__class__.__name__))
            else:
                # set default value here
                if labels is None:
                    labels = np.unique(y_true.values.reshape(-1,)).tolist()
                    if 'labels' in _valid_args(self.process):
                        score_args['labels'] = labels

        # check for using sample_weights in scorer
        if 'sample_weight' in _valid_args(self.process):

            # grab the weights from score_args if possible
            sample_weight = score_args.get('sample_weight', None)

            # check if the sample_weight_attr exists in Score parameters
            if self._sample_weight_attr is not None:
                if sample_weight is not None:
                    warnings.warn(
                        "Both sample_weight_attr and sample_weights were provided, %s is overriding sample_weights"
                        " with the weights implied by sample_weight_attr!" % (self.__class__.__name__,))

                    # pop from original args
                    score_args.pop('sample_weight')
                sample_weight = ds.metadata[self._sample_weight_attr]
            else:
                # check if sample weight is None
                if sample_weight is None:
                    sample_weight = Series(index=ds_new.metadata.index, data=[1.0]*ds_new.n_samples)
                else:
                    # check for the correct length
                    if len(sample_weight) != ds.n_samples:
                        raise ValueError("The number of entries in sample_weight must equal the number of samples"
                                         " in your dataset!")
                    else:
                        # make into series
                        sample_weight = Series(index=ds.metadata.index, data=sample_weight).loc[ds_new.metadata.index]

                    # pop from original args
                    score_args.pop('sample_weight')
        else:
            sample_weight = None

        # compute scores
        if self.verbosity > 0:
            print("Scoring predictions with %s..." % (self.process_name,))

        for i, tvt in enumerate(score_rows):

            # compute appropriate indices
            ids = (tvt_labels == tvt)

            # generate sample weights on slice
            if sample_weight is not None:
                score_args['sample_weight'] = sample_weight.loc[ids].values

            # compute score
            score = scorer(y_true.loc[ids], y_pred.loc[ids], **score_args)

            # check type of score an retype scores as needed
            if i == 0:
                if not (type(score) in [float, int, str]):
                    scores = scores.astype(object)

            # try to store the score (fingers crossed!)
            try:
                scores.loc[tvt] = score
            except ValueError:
                scores[tvt] = score

        # update result
        pred_type_prefix = pred_type.split('_')[0]
        result['_'.join([pred_type_prefix, 'pred_scores'])] = scores

        # print result
        if self.verbosity > 1:
            print("\n%s scores:" % (scores.name))
            for row in scores.index:
                score = scores.loc[row]
                if pd.api.types.infer_dtype(score) == "floating":
                    print("%s: %.2f%%" % (row, score * 100))
                else:
                    print("%s:\n%s" % (row, str(score)))
        return result

    def run(self, ds: DataSet, batch_args: dict = None) -> Tuple[DataSet, dict]:

        # run super
        ds, results = super(Score, self).run(ds, batch_args)

        # print mean, std, min, max
        if self.verbosity > 1:
            # collapse scores first
            scores = self._collapse_class_pred_scores()

            # check the dtype
            try:
                if type(np.array(scores).reshape(-1,)[0:1].item()) == float:
                    levels = ['\d+', '\d+_\d+']
                    for level in levels:
                        # compute first level scores
                        fl_scores = scores.filter(regex='batch_' + level)
                        if fl_scores.size > 0:
                            valid_score_type = ~fl_scores.apply(lambda x: pd.unique(x)[0], axis=1).isna()
                            fl_scores = fl_scores.loc[valid_score_type.values]
                            print("Batches %s:" % ('/'.join(fl_scores.index.tolist())))
                            for score_type in fl_scores.index:
                                print("\t%s:" % (score_type,))
                                for stat, score in self._compute_stats(fl_scores.loc[score_type]).items():
                                    print("\t%s: %.2f%%" % (stat, score * 100))
                                print()
                            print()
            except AttributeError:
                pass


        return ds, results

    def condense_scores(self) -> dict:
        """Condenses scores a dictionary of dataframes which contains scores for split types."""
        
        # generate scores
        scores = self._collapse_class_pred_scores()

        # generate levels
        levels = ['\d+', '\d+_\d+']

        # initialize output
        rep = dict()

        for level in levels:
            # compute first level scores
            fl_scores = scores.filter(regex='batch_' + level)
            fl_scores = fl_scores.dropna()

            # fill in dataframe
            if level == '\d+':
                level_type = "train_test"
            elif level == '\d+_\d+':
                level_type = "train_valid_test"
            rep[level_type] = fl_scores.transpose()
                                        
        return rep

    def _process_classification_labels(self, score_process: Callable) -> Callable:
        """
        Private method to generate a scoring metric on classification labels which
        contains label information.

        Args:
            score_process (Callable): Scoring metric to use on classification labels.

        Returns:
            Callable : Decorated function which will return a score with appropriate labeling
        """

        def class_labels_scorer(y_true, y_pred, **kwargs):

            # apply plain scorer
            score = score_process(y_true.values.reshape(-1,), y_pred.values.reshape(-1,), **kwargs)

            # grab labels
            labels = kwargs.get('labels', [])

            # format the score with labels
            score = self._format_ndarray_output_with_labels(score, labels)

            return score

        return class_labels_scorer

    def _process_classification_scores(self, score_process: Callable) -> Callable:
        """
        Private method to generate a scoring metric on classification labels which
        contains label information.

        Args:
            score_process (Callable): Scoring metric to use on classification labels.

        Returns:
            Callable : Decorated function which will return a score with appropriate labeling
        """
        def class_scores_scorer(y_true, y_pred, **kwargs):
            """
            Private method to generate a scoring metric on classification scores which
            contains label information.

            Args:
                score_process (Callable): Scoring metric to use on classification scores.

            Returns:
                Callable : Decorated function which will return a score with appropriate labeling
            """

            # change shape of y_true to mimic scores (one-hot-encoding)
            y_true_reformated = DataFrame(index=y_true.index,
                                          data=np.array([(y_true == col).astype(int).values.reshape(-1,)
                                                         for col in y_pred.columns]).transpose(),
                                          columns=y_pred.columns)
            # grab labels
            labels = kwargs.get('labels', [])

            if 'labels' in _valid_args(score_process) or labels == []:
                # apply plain scorer
                score = score_process(y_true_reformated.values, y_pred.values, **kwargs)
            else:
                new_args = deepcopy(kwargs)
                new_args.pop('labels')
                y_true_reformated = y_true_reformated[labels]
                y_pred = y_pred[labels]
                score = score_process(y_true_reformated.values, y_pred.values, **new_args)

            # format the score with labels
            score = self._format_ndarray_output_with_labels(score, labels)

            return score

        return class_scores_scorer

    def _process_regression_values(self, score_process: Callable) -> Callable:
        """
        Private method to generate a scoring metric on regression scores which
        contains label information.

        Args:
            score_process (Callable): Scoring metric to use on regression scores.

        Returns:
            Callable : Decorated function which will return a score with appropriate labeling
        """
        pass

    def _collapse_class_pred_scores(self) -> DataFrame:
        """
        Collapses ``class_pred_scores`` into a dataframe where the columns are given as batches and the index is given as
        the sample tvt split, either "Train", "Valid", or "Test".

        Returns:
            DataFrame : Collapsed scores of classification results

        """
        # collapse the dict of dict of dataframe to dataframe
        results = _collapse_dict_dict_pandas(kv=self.results_,
                                             inner_key="class_pred_scores")
        return results

    def _format_ndarray_output_with_labels(self, score, labels: list) -> Union[Series, DataFrame]:
        """
        Private method used to infer labels on output score from :py:attr:`Score.process`.

        Args:
            score (object): The score to be adorned with label information.
            labels (list): Labels to do the adorning with.

        Returns:
            Dataframe or Series : Score adorned with relevant labels.
        """
        # check if score is array-like
        if isinstance(score, (list, tuple, ndarray)):

            score = np.array(score)  # convert to ndarray for consistency

            # handle 1-d case
            if len(score.shape) in [1, 2]:

                # check length of output compared to labels
                if score.shape[0] == len(labels):

                    # apply labels?
                    if self._infer_class_labels_on_output:
                        index = labels
                    else:
                        index = None
                else:
                    index = None

                if len(score.shape) == 2:
                    if score.shape[1] == len(labels):
                        # apply labels?
                        if self._infer_class_labels_on_output:
                            columns = labels
                        else:
                            columns = None
                    else:
                        columns = None

                    # create dataframe of score
                    score = DataFrame(data=score, index=index, columns=columns)
                else:
                    # create series of score
                    score = Series(data=score, index=index)

        return score

    def _compute_stats(self, scores: Series) -> dict:
        """
        Private method used to compute basic statistics of scores on classification/regression results.

        Args:
            scores (Series): Scores across batches.
        Returns:
            dict : mean, standard deviation, minimum, and maximum score across batches.
        """
        mean_score = scores.mean(skipna=True)
        std_score = scores.std(skipna=True)
        min_score = scores.min(skipna=True)
        max_score = scores.max(skipna=True)

        stats = {"Mean": mean_score,
                 "Std. Dev.": std_score,
                 "Minimum": min_score,
                 "Maximum": max_score}

        return stats


class Pipeline(Process):
    """
    :py:class:`Process` subclass used create a seemless pipeline of processes. The :py:class:`Pipeline` class
    acheives the following:

    * Processes run sequantially
    * Results from previous processes are passed/inherited along the way.
    * The pipeline can be saved along the way as to create checkpoints.
    * The pipeline can be run to a certain point and then can continue from that point at a later time.

    Parameters:
        processes (tuple of Process): Contains the processes in the order in which they are meant to be run.

        pipeline_name (str): The common name assigned to the :py:class:`Pipeline` instance.

        parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
            Default is None. :py:func:`ray.init` must be called to initiate the ray cluster before any running
            can be done. If provided, the ``parallel`` value set here will be assigned to each process within.

        verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
            the verbosity the larger the text output. Default is None, indicating the standard text output with a
            :py:class:`Process` instance. If provided, the ``verbosity`` set here will be assigned to each process within.

        checkpoint_path (str): File path indicating the location of the saved, or to be saved, pipeline.
            Default is None.

    Attributes:
        processes (tuple of Process): Contains the processes in the order in which they are meant to be run.

        pipeline_name (str): The common name assigned to the :py:class:`Pipeline` instance.

        parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
            Default is False. :py:func:`ray.init` must be called to initiate the ray cluster before any running
            can be done.

        verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
            the verbosity the larger the text output. Default is 1, indicating the standard text output with a
            :py:class:`Process` instance.

        checkpoint_path (str): File path indicating the location of the saved, or to be saved, pipeline.
            Default is None.

        run_status_ (int): Indicates whether or not the process has finished. A value of 0 indicates the process has not
            finished, a value of 1 indicated the pipeline has finished.

        results_ (dict of dicts): The results of the run process. The keys of the dictionary indicates the batch results
            for a given batch. For each batch there is a dictionary of results with keys indicating the result type, e.g,
            training/validation/test labels (``tvt_labels``), classification labels (``class_labels``), etc...
            A :py:class:`Pipeline` instance, after it runs, outputs any of the results generated by its processes
            contained in :py:attr:`Pipeline.processes`. Refer to each individual process's docstring for a description
            of its results.

    Examples:
            >>> # imports
            >>> import os
            >>> import numpy as np
            >>> from orthrus.core.pipeline import *
            >>> from sklearn.ensemble import RandomForestClassifier as RFC
            >>> from sklearn.model_selection import StratifiedShuffleSplit, KFold
            >>> from sklearn.preprocessing import FunctionTransformer
            >>> from sklearn.metrics import balanced_accuracy_score
            >>> from orthrus.core.dataset import load_dataset
            ...
            >>> # load dataset
            >>> ds = load_dataset(os.path.join(os.environ['ORTHRUS_PATH'],
            ...                                'test_data/Iris/Data/iris.ds'))
            ...
            >>> # define 80-20 train/test partition
            >>> shuffle = Partition(process=StratifiedShuffleSplit(n_splits=1,
            ...                                                    random_state=113,
            ...                                                    train_size=.8),
            ...                     process_name='80-20-tr-tst',
            ...                     split_attr ='species',
            ...                     )
            ...
            >>> # define 5-fold partition for train/valid/test
            >>> kfold = Partition(process=KFold(n_splits=5,
            ...                                 shuffle=True,
            ...                                 random_state=124,
            ...                                 ),
            ...                   process_name='5-fold-CV')
            ...
            >>> # define log transform process
            >>> log = Transform(process=FunctionTransformer(np.log),
            ...                 process_name='log',
            ...                 retain_f_ids=True)
            ...
            >>> # define random forest classify process
            >>> rf = Classify(process=RFC(),
            ...                process_name='RF',
            ...                class_attr='species')
            ...
            >>> # define balance accuracy score process
            >>> bsr = Score(process=balanced_accuracy_score,
            ...             process_name='bsr',
            ...             pred_attr='species')
            ...
            >>> # define the pipeline
            >>> pipeline = Pipeline(processes=(log,
            ...                                shuffle,
            ...                                kfold,
            ...                                rf,
            ...                                bsr),
            ...                     pipeline_name='example',
            ...                     checkpoint_path=os.path.join(os.environ['ORTHRUS_PATH'],
            ...                                                  'test_data/Iris/example_pipeline.pickle'),
            ...                     verbosity=2)
            ...
            >>> # run the pipeline
            >>> ds, results = pipeline.run(ds)
            -------------------
            Batches Train/Test:
                Train:
                Mean: 100.00%
                Std. Dev.: 0.00%
                Minimum: 100.00%
                Maximum: 100.00%
                _
                Test:
                Mean: 95.94%
                Std. Dev.: 3.98%
                Minimum: 90.91%
                Maximum: 100.00%
            -------------------------
            Batches Train/Test/Valid:
                Train:
                Mean: 100.00%
                Std. Dev.: 0.00%
                Minimum: 100.00%
                Maximum: 100.00%
                _
                Test:
                Mean: 97.33%
                Std. Dev.: 1.49%
                Minimum: 96.67%
                Maximum: 100.00%
                _
                Valid:
                Mean: 93.88%
                Std. Dev.: 4.75%
                Minimum: 87.50%
                Maximum: 100.00%

            >>> # define the pipeline
            >>> pipeline = Pipeline(processes=(log,
            ...                                shuffle,
            ...                                kfold,
            ...                                rf,
            ...                                bsr),
            ...                     pipeline_name='example',
            ...                     checkpoint_path=os.path.join(os.environ['ORTHRUS_PATH'],
            ...                                                  'test_data/Iris/example_pipeline.pickle'),
            ...                     verbosity=2)
            ...
            >>> # run the pipeline with checkpoint, stop before rf
            >>> ds, results = pipeline.run(ds, checkpoint=True, stop_before='RF')
            ...
            >>> # simulate stop and reloading
            >>> del pipeline
            >>> pipeline = Pipeline(checkpoint_path=os.path.join(os.environ['ORTHRUS_PATH'],
            ...                                                  'test_data/Iris/example_pipeline.pickle'))
            ...
            >>> # finish the pipeline
            >>> pipeline.run(ds)
            ---------------------
            Starting 0th process log...
            _
            Saving current state of pipeline to disk...
            _
            Starting 1th process 80-20-tr-tst...
            _
            Saving current state of pipeline to disk...
            _
            Starting 2th process 5-fold-CV...
            _
            Saving current state of pipeline to disk...
            _
            Loading Pipeline example from file...
            _
            Starting Pipeline example from process RF...
            _
            Starting 3th process RF...
            _
            Starting 4th process bsr...
            Batches Train/Test:
                Train:
                Mean: 100.00%
                Std. Dev.: nan%
                Minimum: 100.00%
                Maximum: 100.00%
                _
                Test:
                Mean: 96.67%
                Std. Dev.: nan%
                Minimum: 96.67%
                Maximum: 96.67%
            -------------------------
            Batches Train/Test/Valid:
                Train:
                Mean: 100.00%
                Std. Dev.: 0.00%
                Minimum: 100.00%
                Maximum: 100.00%
                _
                Test:
                Mean: 97.33%
                Std. Dev.: 1.49%
                Minimum: 96.67%
                Maximum: 100.00%
                _
                Valid:
                Mean: 93.36%
                Std. Dev.: 5.14%
                Minimum: 87.50%
                Maximum: 100.00%
    """
    def __init__(self,
                 processes: Tuple[Process, ...] = tuple(),
                 pipeline_name: str = None,
                 parallel: bool = None,
                 verbosity: int = None,
                 checkpoint_path: str = None,
                 ):

        # parameters
        self.pipeline_name = pipeline_name
        self.processes = processes

        # set and parallel and verbosity globally if True
        if parallel is not None:
            for process in processes:
                process.parallel = parallel

        if verbosity is not None:
            for process in processes:
                process.verbosity = verbosity
        else:
            verbosity = 1

        # init with Process class
        super(Pipeline, self).__init__(process=None,
                                       process_name=None,
                                       parallel=parallel,
                                       verbosity=verbosity,
                                       )

        # private attributes
        self._current_process = 0
        self._checkpoint_path = checkpoint_path
        self._stop_before = None
        self._labels = ["pipeline_name", "parallel", "verbosity", "checkpoint_path"]

        # try to load self from checkpoint path
        if self.checkpoint_path is not None:
            checkpoint = load_object(self.checkpoint_path, block=False)
            if checkpoint is None:
                warnings.warn("No Pipeline found at checkpoint_path! The Pipeline will start from the beginning and"
                              " use the checkpoint_path provided to save instances of the Pipeline.")
            else:
                if self.verbosity > 0:
                    print("Loading Pipeline %s from file..." % (checkpoint.pipeline_name,))
                # save these parameters
                checkpoint_path = deepcopy(self._checkpoint_path)
                stop_before = self._stop_before

                # update params from saved pipeline
                self.__dict__.update(checkpoint.__dict__)

                # set these back to the original pipeline
                self._checkpoint_path = checkpoint_path
                self._stop_before = stop_before

    def __repr__(self):
        params = inspect.signature(self.__init__).parameters
        pipeline_str = '(' + ', '.join([process.process_name for process in self.processes]) + ')'
        non_default_labels = [label for label in self._labels if getattr(self, label) != params[label].default]
        kws = [f"{key}={getattr(self, key)!r}" for key in non_default_labels]
        kws.insert(0, "processes=" + pipeline_str)
        return "{}({})".format(type(self).__name__, ", ".join(kws))

    @property
    def process_name(self) -> str:
        """
        Gives the name of the current process.
        """
        return self.processes[self._current_process].process_name

    @property
    def checkpoint_path(self) -> str:
        """
        Generates checkpoint path for loading a pipeline from a pickle file.
        """
        if self._checkpoint_path is None:
            return None
        else:
            return os.path.abspath(self._checkpoint_path)

    @checkpoint_path.setter
    def checkpoint_path(self, checkpoint_path: str):
        self._checkpoint_path = checkpoint_path

    @property
    def stop_before(self) -> int:
        """
        Generates integer index for process to stop before in pipeline.
        """
        # grab stop_before value
        sb = self._stop_before

        if sb is None:
            idx = len(self.processes)

        elif type(sb) == str:
            hit = [process.process_name == sb for process in self.processes]
            ids = np.where(hit)[0]
            if ids.size > 1:
                raise ValueError("Non-unique identifier for stop_before process, "
                                 "make the names of your processes unique if you are using a string identifier!")
            elif ids.size == 0:
                raise ValueError("Identifier for stop_before process does not exist in processes!")
            else:
                idx = ids.item()

        elif type(sb) == int:
            idx = sb

        else:
            raise ValueError("stop_before must be either an integer or"
                             " a string which identifies the process to stop before!")

        # check for appropriate index
        if idx > len(self.processes) - 1:
            return None
        elif idx <= self._current_process:
            raise ValueError("Integer index for stop_before process must be"
                             " strictly greater than self._current_process!")

        return idx

    def _run(self, ds: DataSet, **kwargs):
        pass

    def run(self,
            ds: DataSet,
            batch_args: dict = None,
            stop_before: Union[str, int] = None,
            checkpoint: bool = False) -> Tuple[DataSet, dict]:

        """
        Runs the pipeline in sequence. The pipeline can be stopped and restarted at a checkpoint.

        Args:
            ds (DataSet): The dataset to process.

            batch_args (dict): A dictionary with keys given by batch. Each value in the dictionary is a dictionary of
                keyword arguments to a sub-classes ``_run`` method. A keyword argument may indicate the training/test
                labels for that batch, or classification labels for that batch, or a batch-specific transform to apply
                to :py:attr:`ds`. Note: Batches should be specified by ``batch_0``, ``batch_1``, ... , ``batch_0_0``,
                ``batch_0_0``, etc if you want to link your processes in a :py:attr:`Pipeline` instance,
                In particular ``batch_0_1`` is considered a derivative batch of ``batch_0`` and will inherit if
                possible batch specific transforms, labels, etc... from ``batch_0``.

            stop_before (int or str): Specifies the process to stop at, for example if a process has the name "fire"
                specifying :py:attr:`stop_before` = "fire" will cause the pipeline to stop before the fire process is
                executed. If the process named "fire" is 3rd in the list of processes then you can simply pass
                :py:attr:`stop_before` = 3. The default is None, and will cause the pipeline to run all the way
                through.

            checkpoint (bool): If ``True`` then the pipeline will save to :py:attr:`Pipeline.checkpoint_path`.
                :py:attr:`Pipeline.checkpoint_path` must be filled in order to use checkpointing! If ``False``
                the pipelin will execute without saving along the way.

        Returns:
            Tuple[DataSet, dict] : The first argument is the object :py:attr:`ds` and the second argument is
            :py:attr:`Process.results_`

        """

        # check if pipeline is finished running
        if self._current_process == len(self.processes):
            if self.verbosity > 0:
                print("Pipeline %s already finished running!" % (self.pipeline_name,))
            return ds, self.results_

        # store stop_before point
        self._stop_before = stop_before

        # check for valid checkpoint
        if checkpoint and self.checkpoint_path is None:
            raise ValueError("No checkpoint_path provided to save instances!")

        # check for where the pipeline is at
        if self._current_process > 0:
            if self.verbosity > 0:
                print("Starting Pipeline %s from process %s..." % (self.pipeline_name,
                                                                   self.processes[self._current_process].process_name,))
        else:
            # make direct reference
            self.results_ = deepcopy(batch_args)

        # define processes that need to complete
        processes = self.processes[self._current_process: self.stop_before]

        # maybe it just works?
        for process in processes:

            # print to screen
            if self.verbosity > 0:
                print("Starting %dth process %s...\n" % (self._current_process, self.process_name))

            # run process
            _, next_results = process.run(ds, self.results_)

            # check for None type initial results
            if self.results_ is None:
                self.results_ = {}

            # compose transforms and update the rest
            self._update_results(self.results_, next_results)

            # check for multiple batches and left over signle batch
            if len(self.results_) > 1 and 'batch' in self.results_.keys():
                del self.results_['batch']

            # update current process
            self._current_process += 1

            # checkpoint if true
            if checkpoint:
                if self.verbosity > 0:
                    print("Saving current state of pipeline to disk...\n")
                self.save(self._checkpoint_path, overwrite=True)

            # debug
            # if self._current_process == 2:
            #     break

        # check if last process
        if self._current_process == len(self.processes):
            self.run_status_ = 1

        return ds, self.results_

    @staticmethod
    def _update_result(result: dict, next_result: dict):
        """
        Private method used to update the result in batch from process to process.

        Args:
            result (dict): The current result on a batch from the current state of the pipeline.
            next_result (dict): The next result on a batch from the next process.

        Returns:
            inplace
        """
        # check if result has transforms
        transforms = result.get('transforms', None)

        # try to extract transform from next_result
        next_transform = next_result.get('transform', None)

        # set default else append
        if next_transform is not None:
            if transforms is None:
                result['transforms'] = (next_transform,)
            else:
                transforms = list(transforms)
                transforms.append(next_transform)
                result['transforms'] = tuple(transforms)

            # set transform to composition
            result['transform'] = compose(result['transforms'])

        # overwrite everything except transform, transformer
        for (k, v) in next_result.items():
            if k not in ['transform', 'transformer']:
                result[k] = v

    def _update_results(self, results, next_results):
        """
        Private method used to update the results from process to process. Calls
        :py:meth:`Pipeline._update_result` internally.

        Args:
            result (dict of dict): The current results from the current state of the pipeline.
            next_result (dict of dict): The next results from the next process.

        Returns:
            inplace
        """
        for batch in next_results.keys():

            # define the next result
            next_result = next_results[batch]

            # check if the batch is in the results
            if batch in results.keys():

                # define the result
                result = results[batch]

                # update the result from the next_result
                self._update_result(result, next_result)

            else:
                # check if it is a sub-batch of another
                super_batch = self._find_super_batch(batch, results)

                # inherit from super batch if the key isn't present
                if super_batch is None:
                    # if not just update the results dict
                    results[batch] = dict()
                    result = results[batch]
                    self._update_result(result, next_result)

                else:
                    # inherit from super batch
                    results[batch] = copy(results[super_batch])
                    result = results[batch]

                    # update result from next result
                    self._update_result(result, next_result)


class Report(Score):
    """
    :py:class:`Process` subclass used to generate a classification report. See sklearn classification_report.

    Parameters:
        parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
            Default is False. :py:func:`ray.init` must be called to initiate the ray cluster before any running
            can be done.

        verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
            the verbosity the larger the text output. Default is 1, indicating the standard text output with a
            :py:class:`Process` instance.

        score_args (dict): Keyword arguments passed to :py:attr:`Score.process()`.

        pred_type (str): Can be either "class_labels", "class_scores", "reg_scores", currently. It indicates the
            type predictions made, i.e., classification labels, classification scores, or regression scores. The
            default is "class_labels".

        sample_weight_attr (str): Attribute in the metadata of the dataset you wish to weight the scores by, e.g.,
            misclassifying a sick sample might be more costly than misclassifying a healthy sample.

        infer_class_labels_on_output (bool): If ``True`` the process will attempt to assign labels to the output score.
            For example if one uses a confusion matrix the process will attempt to assign the class labels given in
            :py:attr:`Score.classes` to the rows and columns for indexing.

        classes (list): Classes used for classification labels. You can provide a subset of classification labels
            to look at scores relative to fewer classes. The default is None.

    Attributes:
         parallel (bool): Flag indicating whether or not to use `ray <https://ray.io/>`_'s parallel processing.
            Default is False. :py:func:`ray.init` must be called to initiate the ray cluster before any running
            can be done.

        verbosity (int): Number indicating the level of verbosity, i.e., text output to console to the user. The higher
            the verbosity the larger the text output. Default is 1, indicating the standard text output with a
            :py:class:`Process` instance.


        pred_type (str): Can be either "class_labels", "class_scores", "reg_scores", currently. It indicates the
            type predictions made, i.e., classification labels, classification scores, or regression scores. The
            default is "class_labels".

        _sample_weight_attr (str): Attribute in the metadata of the dataset you wish to weight the scores by, e.g.,
            misclassifying a sick sample might be more costly than misclassifying a healthy sample.

        _infer_class_labels_on_output (bool): If ``True`` the process will attempt to assign labels to the output score.
            For example if one uses a confusion matrix the process will attempt to assign the class labels given in
            :py:attr:`Score.classes` to the rows and columns for indexing.

        _classes (list): Classes used for classification labels. You can provide a subset of classification labels
            to look at scores relative to fewer classes. The default is None.

        run_status_ (int): Indicates whether or not the process has finished. A value of 0 indicates the process has not
            finished, a value of 1 indicated the process has finished.

        results_ (dict of dicts): The results of the run process. The keys of the dictionary indicates the batch results
            for a given batch. For each batch there is a dictionary of results with keys indicating the result type, e.g,
            training/validation/test labels (``tvt_labels``), classification labels (``class_labels``), etc...
            A :py:class:`Score` instance, after it runs, outputs the following results per batch:

            * class_pred_scores (Series): Scores generated by :py:attr:`Score.process` on classification results
              generated by :py:class:`Classify`. The index of the ``Series`` is given by the labels in
              ``batch['tvt_labels']``, e.g., "Train", "Valid", "Test". The values of the ``Series`` are the associated
              scores for each sample type: "Train", "Valid", "Test".

            * reg_pred_scores (Series): Scores generated by :py:attr:`Score.process` on regression results
              generated by :py:class:`Regress`. The index of the ``Series`` is given by the labels in
              ``batch['tvt_labels']``, e.g., "Train", "Valid", "Test". The values of the ``Series`` are the associated
              scores for each sample type: "Train", "Valid", "Test".

    Examples:
            >>> # imports
            >>> import os
            >>> from orthrus.core.pipeline import Report, Classify, Partition
            >>> from sklearn.ensemble import RandomForestClassifier as RFC
            >>> from sklearn.model_selection import StratifiedShuffleSplit
            >>> from orthrus.core.dataset import load_dataset
            ...
            >>> # load dataset
            >>> ds = load_dataset(os.path.join(os.environ['ORTHRUS_PATH'],
            ...                            'test_data/Iris/Data/iris.ds'))
            ...
            >>> # define 80-20 train/test partition
            >>> shuffle = Partition(process=StratifiedShuffleSplit(n_splits=1,
            ...                                                random_state=113,
            ...                                                train_size=.8),
            ...                    process_name='80-20-tr-tst',
            ...                    verbosity=1,
            ...                    split_attr='species',
            ...                    )
            ...
            >>> # define random forest classify process
            >>> rf = Classify(process=RFC(),
            ...            process_name='RF',
            ...            class_attr='species',
            ...            verbosity=1)

            >>> # define balance accuracy score process
            >>> report = Report(pred_attr='species',
            ...                verbosity=2)
            ...
            >>> # run partition and classification processes
            >>> ds, results_0 = shuffle.run(ds)
            >>> ds, results_1 = rf.run(ds, results_0)
            ...
            >>> # carry over tvt_labels, use Pipeline for chaining processes instead!
            >>> [results_1[batch].update(results_0[batch]) for batch in results_1]
            ...
            >>> # score classification results
            >>> ds, results = report.run(ds, results_1)

            Now we can plot the statistics of our report. For example we will plot the test scores
            attained using our random forest classifier.

            >>> # imports
            >>> from matplotlib import pyplot as plt
            >>> import numpy as np
            ...
            >>> # plot test scores
            >>> test_scores = report.report()['train_test'].filter(regex="^((?!Support).)*$").filter(regex="Test")
            >>> test_scores.columns = test_scores.columns.str.strip("Test_")
            >>> test_scores.loc["batch_0_report_scores"].plot.bar(title="Iris Dataset Random Forest Test Scores",
            ...                                                   rot=30, figsize=(15, 10), grid=True,
            ...                                                   yticks=np.arange(0, 1.1, .1))
            >>> plt.savefig(os.path.join(os.environ['ORTHRUS_PATH'], "docsrc/figures/iris_rf_test_scores.png"))

            .. figure:: ../figures/iris_rf_test_scores.png
                :width: 800px
                :align: center
                :alt: alternate text
                :figclass: align-center
    """

    def __init__(self,
                 pred_attr: Union[str, list],
                 parallel: bool = False,
                 verbosity: int = 1,
                 pred_type: str = 'class_labels',
                 sample_weight_attr: str = None,
                 infer_class_labels_on_output: bool = True,
                 classes: list = None,
                 process=None,
                 process_name=None,
                 ):

        # init with Process class
        super(Score, self).__init__(process=classification_report,
                                    process_name="report",
                                    parallel=parallel,
                                    verbosity=verbosity,
                                    )
        # parameters
        self.pred_type = pred_type
        self.pred_attr = pred_attr
        self.score_args = dict(output_dict=True)

        # private attributes
        self._classes = classes
        self._sample_weight_attr = sample_weight_attr
        self._infer_class_labels_on_output = infer_class_labels_on_output
        self._labels += ["pred_type", "pred_attr", "score_args"]

    def report(self):

        # generate scores
        scores = self._collapse_class_pred_scores()

        # generate levels
        levels = ['\d+', '\d+_\d+']

        # initialize output
        rep = dict()

        # extract score names
        score = scores.iloc[0,0]
        score_prefix = score.index
        score_dict = dict()
        for key in score_prefix:
            try:
                score_dict[key] = list(score[key].keys())
            except AttributeError:
                score_dict[key] = []

        for level in levels:
            # compute first level scores
            fl_scores = scores.filter(regex='batch_' + level)
            fl_scores = fl_scores.dropna()

            # fill in dataframe
            if level == '\d+':
                level_type = "train_test"
            elif level == '\d+_\d+':
                level_type = "train_valid_test"
            rep[level_type] = pd.DataFrame(index=fl_scores.columns)

            # loop through scores
            for score_type in fl_scores.index:

                # grab type of scores
                s = fl_scores.loc[score_type]

                for key in score_dict.keys():
                    if score_dict[key] == []:
                        # fill in columns
                        col = ":".join([score_type.capitalize(), key])
                        rep[level_type][col] = pd.NA
                        for batch in fl_scores.columns:
                            try:
                                rep[level_type].loc[batch, col] = s[batch][key]
                            except KeyError:
                                pass
                    else:
                        for metric in score_dict[key]:

                            # fill in columns
                            col = ":".join([score_type.capitalize(), key, metric.capitalize()])
                            rep[level_type][col] = pd.NA
                            for batch in fl_scores.columns:
                                try:
                                    rep[level_type].loc[batch, col] = s[batch][key][metric]
                                except KeyError:
                                    pass
                                
        return rep

# TODO: Add Regress Class

# TODO: Add _collapse_reg_pred_scores in Score

# TODO: Implement Tune(Process) using ray

# TODO: Implement batch restriction, .e.g, Process.run(..., batches:list = [...])

# TODO: Implement parallel pipelines
