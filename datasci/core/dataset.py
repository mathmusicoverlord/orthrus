"""
This module contains the main DataSet class used for all preprocessing, visualization, feature selection,
and classification.
"""

# imports
import os
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
from numpy.core import ndarray
from pandas.core.frame import DataFrame
from sklearn.preprocessing import FunctionTransformer
from pandas.core.frame import Series
from datasci.core.helper import scatter_pyplot
from datasci.core.helper import scatter_plotly

# classes
class DataSet:
    """
    Primary base class for storing data and metadata for a generic dataset. Contains methods for quick data
    pre-processing, visualization, and classification.

    Parameters:
        name (str): Reference name for the dataset. Default is the empty string.

        description (str): Short description of data set.

        path (str): File path for saving DataSet instance and related outputs. Default is the empty string.

        data (pandas.DataFrame): Numerical data or features of the data set arranged as samples x features.
            Default is the empty DataFrame.

        metadata (pandas.DataFrame): Categorical data or attributes of the dataset arranged as samples x attributes.
            The sample labels in the index column should be the same format as those used for the data DataFrame.
            If labels are missing or there are more labels than in the data, the class will automatically restrict
            to just those samples used in the data and fill in NaN where there are missing samples. Default is the
            empty DataFrame.

        vardata (pandas.DataFrame): Categorical data or attributes of the features on the dataset arranged as
            features x attributes. The feature labels in the index column should be the same format as those used for
            the columns in the data DataFrame. Default is None.

        dissimilarity_matrix (pandas.DataFrame): Symmetric matrix whose columns and index are given by the samples.
            Its contents give the pairwise dissimilarities between the samples. Default is None.

        normalization_method (str): Label indicating the normalization used on the data. Future normalization will
            append as normalization_1/normalization_2/.../normalization_n indicating the sequence of normalizations
            used on the data. Default is the empty string.

        imputation_method (str): Label indicating the imputation used on the data. Default is the empty string.

    Attributes:
        name (str): Reference name for the dataset. Default is the empty string.

        description (str): Short description of data set.

        path (str): File path for saving DataSet instance and related outputs. Default is the empty string.

        data (pandas.DataFrame): Numerical data or features of the data set arranged as samples x features.
            Default is the empty DataFrame.

        metadata (pandas.DataFrame): Categorical data or attributes of the dataset arranged as samples x attributes.
            The sample labels in the index column should be the same format as those used for the data DataFrame.
            If labels are missing or there are more labels than in the data, the class will automatically restrict
            to just those samples used in the data and fill in NaN where there are missing samples. Default is the
            empty DataFrame.

        vardata (pandas.DataFrame): Categorical data or attributes of the features on the dataset arranged as
            features x attributes. The feature labels in the index column should be the same format as those used for
            the columns in the data DataFrame. Default is None.

        dissimilarity_matrix (pandas.DataFrame): Symmetric matrix whose columns and index are given by the samples.
            Its contents give the pairwise dissimilarities between the samples. Default is None.

        normalization_method (str): Label indicating the normalization used on the data. Future normalization will
            append as normalization_1/normalization_2/.../normalization_n indicating the sequence of normalizations
            used on the data. Default is the empty string.

        imputation_method (str): Label indicating the imputation used on the data. Default is the empty string.

        experiments (dict): Holds experimental results. e.g. from :py:meth:`DataSet.classify`.


    Examples:
            >>> from pydataset import data as pydat
            >>> from datasci.core.dataset import DataSet as DS
            >>> df = pydat('iris')
            >>> data = df[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
            >>> metadata = df[['Species']]
            >>> ds = DS(name='Iris', data=data, metadata=metadata)
    """

    # methods
    def __init__(self, name: str = '',
                 description: str = '',
                 path: str = os.curdir,
                 data: DataFrame = pd.DataFrame(),
                 metadata: DataFrame = pd.DataFrame(),
                 vardata: DataFrame = None,
                 dissimilarity_matrix: DataFrame = None,
                 normalization_method: str = '',
                 imputation_method: str = ''):

        # Load attributes
        self.name = name
        self._description = description
        self.path = path.rstrip('/') + '/'
        self.data = data
        self.metadata = metadata
        self.normalization_method = normalization_method
        self.imputation_method = imputation_method

        # unset attributes
        self.experiments = dict()

        # restrict metadata to data
        try:
            meta_index = self.metadata.index.intersection(self.data.index)
            missing_data_index = self.data.index.drop(meta_index)
            missing_data = pd.DataFrame(index=missing_data_index, columns=self.metadata.columns)
        except AttributeError:
            data_index = self.data.index.to_pandas()
            meta_index = self.metadata.index.to_pandas().intersection(data_index)
            missing_data_index = data_index.drop(meta_index)
            missing_data = self.metadata.__class__(index=missing_data_index, columns=self.metadata.columns)

        self.metadata = self.metadata.loc[meta_index].append(missing_data)

        # sort data and metadata to be in same order
        self.metadata = self.metadata.loc[self.data.index]

        # Assign vardata
        if vardata is None:
            self.vardata = pd.DataFrame(index=self.data.columns)
        else:
            self.vardata = vardata
            try:
                # restrict vardata to data
                var_index = vardata.index.intersection(self.data.transpose().index)
                missing_data_index = self.data.transpose().index.drop(var_index)
                missing_data = pd.DataFrame(index=missing_data_index, columns=self.vardata.columns)
            except AttributeError:
                # restrict vardata to data
                data_index = self.data.transpose().index.to_pandas()
                var_index = vardata.index.to_pandas().intersection(data_index)
                missing_data_index = data_index.drop(var_index)
                missing_data = self.data.__class__(index=missing_data_index, columns=self.vardata.columns)

            self.vardata = vardata.loc[var_index].append(missing_data)
            self.vardata = self.vardata.loc[self.data.columns]

        # Assign dissimilarity matrix
        if dissimilarity_matrix is None:
            self.dissimilarity_matrix = None
        else:
            try:
                dissimilarity_index = dissimilarity_matrix.index.intersection(self.data.index)
                missing_data_index = self.data.index.drop(dissimilarity_index)
                missing_data = pd.DataFrame(index=missing_data_index)
            except AttributeError:
                data_index = self.data.index.to_pandas()
                dissimilarity_index = dissimilarity_matrix.to_pandas().intersection(data_index)
                missing_data_index = data_index.drop(meta_index)
                missing_data = self.data.__class__(index=missing_data_index)

            self.dissimilarity_matrix = dissimilarity_matrix.loc[dissimilarity_index].append(missing_data)

            # sort data and dissimilarity matrix to be in same order
            self.dissimilarity_matrix = self.dissimilarity_matrix.loc[self.data.index, self.data.index.to_list()]

            # convert index and column names to string
            self.dissimilarity_matrix.index = self.dissimilarity_matrix.index.astype(str)
            self.dissimilarity_matrix.columns = self.dissimilarity_matrix.columns.astype(str)

        # convert all indices and column names to string to avoid integer slicing issues
        self.data.index = self.data.index.astype(str)
        self.metadata.index = self.metadata.index.astype(str)
        self.vardata.index = self.vardata.index.astype(str)
        self.data.columns = self.data.columns.astype(str)
        self.metadata.columns = self.metadata.columns.astype(str)
        self.vardata.columns = self.vardata.columns.astype(str)

    @property
    def n_samples(self) -> int:
        """
        The number of samples in the dataset.

        Returns: The number of samples in the dataset.
        """

        return self.data.shape[0]

    @property
    def n_features(self) -> int:
        """
        The number of features in the dataset.

        Returns: The number of features in the dataset.
        """

        return self.data.shape[1]


    def visualize(self, embedding,
                  attr: str,
                  cross_attr: str = None,
                  feature_ids=None,
                  sample_ids=None,
                  use_dissimilarity: bool = False,
                  backend: str = 'pyplot',
                  viz_name: str = None,
                  supervised: bool = False,
                  save: bool = False,
                  save_name: str = None,
                  **kwargs):
        """
        This method visualizes the data by embedding it in 2 or 3 dimensions via the transformation
        :py:attr:`embedding`. The user can restrict both the sample indices and feature indices, as well
        as color and mark the samples by chosen metadata attributes. The transformation will happen post
        restricting the features and samples.

        Args:
            embedding (object): Class instance which must contain the method fit_transform. The output of
                embedding.fit_transform(:py:attr:`DataSet.data`) must have at most 3 columns.

            attr (str): Name of the metadata attribute to color samples by.

            cross_attr (str): Name of the secondary metadata attribute to mark samples by.

            feature_ids (list-like): List of indicators for the features to use. e.g. [1,3], [True, False, True],
                ['gene1', 'gene3'], etc..., can also be pandas series or numpy array. Defaults to use all features.

            sample_ids (like-like): List of indicators for the samples to use. e.g. [1,3], [True, False, True],
                ['human1', 'human3'], etc..., can also be pandas series or numpy array. Defaults to use all samples.

            use_dissimilarity (bool): If True the embedding will fit to the dissimilarity matrix stored in
                :py:attr:`DataSet.dissimilarity_matrix`. The default is false.

            backend (str): Plotting backend to use. Can be either ``pyplot`` or ``plotly``. The default is ``pyplot``.

            viz_name (str): Common name for the embedding used. e.g. MDS, PCA, UMAP, etc... The default is
                :py:attr:`embedding`.__str__().

            supervised (bool): If True the attr labels are based to the embedding fit method, rather than None.

            save (bool): Flag indicating to save the file. The file will save to self.path with the file name
                :py:attr:`DataSet.name` _ :py:attr:`viz_name` _ :py:attr:`attrname`.png for ``pyplot`` and
                :py:attr:`DataSet.name` _ :py:attr:`viz_name` _ :py:attr:`attrname`.html for ``plotly``

            save_name (str): Optional file name to save figure to when :py:attr:`save` is ``True``. This save name will
            be prepended by :py:attr:`DataSet.path`. Default is None.

            **kwargs (dict): Keyword arguments passed directly to :py:func:`helper.scatter_pyplot` when using the
                backend ``pyplot`` and :py:func:`helper.scatter_plotly` when using the backend ``plotly``, for
                indicating plot properties.

        Returns:
             class instance: The fit embedding used to visualize.

             ndarray of shape (n_samples, n_components): The values of the embedding.

        Examples:
            >>> from pydataset import data as pydat
            >>> from datasci.core.dataset import DataSet as DS
            >>> from sklearn.manifold import MDS
            >>> df = pydat('iris')
            >>> data = df[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
            >>> metadata = df[['Species']]
            >>> ds = DS(name='Iris', data=data, metadata=metadata)
            >>> embedding = MDS(n_components=3)
            >>> ds.visualize(embedding=embedding, attr='Species', no_axes=True)

            >>> from pydataset import data as pydat
            >>> from datasci.core.dataset import DataSet as DS
            >>> from sklearn.decomposition import PCA
            >>> import numpy as np
            >>> df = pydat('iris')
            >>> data = df[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
            >>> metadata = df[['Species']]
            >>> metadata['test'] = np.random.randint(1, 3, (metadata.shape[0],))
            >>> ds = DS(name='Iris', data=data, metadata=metadata)
            >>> embedding = PCA(n_components=3)
            >>> ds.visualize(embedding=embedding,
            ...              attr='Species',
            ...              cross_attr='test',
            ...              xlabel='PC 1',
            ...              ylabel='PC 2',
            ...              zlabel='PC 3',
            ...              backend='plotly',
            ...              mrkr_size=10,
            ...              mrkr_list=['circle', 'cross'],
            ...              figsize=(900,800),
            ...              use_dash=True,
            ...              debug=True,
            ...              save=True)
        """
        # set defaults
        if viz_name is None:
            viz_name = embedding.__str__().split('(')[0]

        # slice the data set
        ds = self.slice_dataset(feature_ids, sample_ids)
        if use_dissimilarity:
            data = ds.dissimilarity_matrix
        else:
            data = ds.data
        metadata = ds.metadata

        # transform data
        if supervised:
            data_trans = embedding.fit_transform(data.values, y=metadata[attr].values)
        else:
            try:
                data_trans = embedding.fit_transform(data.values, y=None)
            except ValueError:
                data_trans = embedding.fit_transform(data.values)

        # store embedding values
        embedding_vals = deepcopy(data_trans)

        if data_trans.shape[1] == 1:
            is_1d = True
            data_trans = np.hstack((data_trans, (2*np.random.rand(data_trans.shape[0], 1) - 1)))
        else:
            is_1d = False

        dim = data_trans.shape[1]

        # create dataframe from transformed data
        data_trans = data.__class__(index=data.index, data=data_trans)
        data_trans = pd.DataFrame(data_trans)

        # restrict metadata to attributes
        if cross_attr is None:
            metadata = metadata[[attr]]
        else:
            metadata = metadata[[attr, cross_attr]]

        # replace nan values
        metadata = deepcopy(metadata)
        #metadata = deepcopy(metadata.fillna('nan'))

        # create common dataframe
        df = data_trans
        if cross_attr is None:
            df[[attr]] = metadata
        else:
            df[[attr, cross_attr]] = metadata

        # generate plot labels
        if self.imputation_method == '':
            imputation_method = 'None'
        else:
            imputation_method = self.imputation_method

        if self.normalization_method == '':
            normalization_method = 'None'
        else:
            normalization_method = self.normalization_method

        if cross_attr is None:
            title = 'Visualization of data set ' + self.name + ' using\n' \
                    + viz_name + ' with labels given by ' + attr
            if save_name is None:
                save_name = os.path.join(self.path, '_'.join([self.name, viz_name, str(imputation_method), str(normalization_method), str(attr), str(dim)]))
            else:
                save_name = os.path.join(self.path, save_name)
        else:
            title = 'Visualization of data set ' + self.name + ' using\n' \
                    + viz_name + ' with labels given by ' + attr + ' and ' + cross_attr
            if save_name is None:
                save_name = os.path.join(self.path, '_'.join([self.name, viz_name, str(imputation_method), str(normalization_method), str(attr),  str(cross_attr), str(dim)]))
            else:
                save_name = os.path.join(self.path, save_name)

        if not save:
            save_name = None

        # set default axis labels
        if dim < 4:
            kwargs['xlabel'] = kwargs.get('xlabel', viz_name + ' ' + str(1))
            if is_1d:
                kwargs['ylabel'] = kwargs.get('ylabel', 'Jitter')
            else:
                kwargs['ylabel'] = kwargs.get('ylabel', viz_name + ' ' + str(2))
            if dim > 2:
                kwargs['zlabel'] = kwargs.get('zlabel', viz_name + ' ' + str(3))


        # plot data
        if backend == "pyplot":
            # set subtitle
            subtitle = 'Imputation: ' + str(imputation_method) + '\nNormalization: ' + str(normalization_method)
            if not ('subtitle' in kwargs.keys()):
                kwargs['subtitle'] = subtitle

            # add to keyword arguments
            if not ('title' in kwargs.keys()):
                kwargs['title'] = title

            scatter_pyplot(df=df,
                        grp_colors=attr,
                        grp_mrkrs=cross_attr,
                        dim=dim,
                        save_name=save_name,
                        **kwargs)

        elif backend == "plotly":
            # set subtitle
            subtitle = 'Imputation: ' + str(imputation_method) + ', Normalization: ' + str(normalization_method)
            if not ('subtitle' in kwargs.keys()):
                kwargs['subtitle'] = subtitle

            # make title fit with html
            title = title.replace('\n', '<br>')

            # add to keyword arguments
            if not ('title' in kwargs.keys()):
                kwargs['title'] = title

            scatter_plotly(df=df,
                        grp_colors=attr,
                        grp_mrkrs=cross_attr,
                        dim=dim,
                        save_name=save_name,
                        **kwargs)

        return embedding, embedding_vals

    def normalize(self,
                normalizer,
                feature_ids=None,
                sample_ids=None,
                norm_name: str = None,
                supervised_attr: str = None,
                normalize_args=None):
        """
        Normalizes the data of the dataset according to a normalizer class. Appends the normalization method
        used to :py:attr:`DataSet.normalization_method`.

        Args:
            normalizer (object): Class instance which must contain the method fit_transform. The output of
                normalizer.fit_transform(:py:attr:`DataSet.data`) must have the same number of columns
                as :py:attr:`DataSet.data`.

            feature_ids (list-like): List of indicators for the features to use. e.g. [1,3], [True, False, True],
                ['gene1', 'gene3'], etc..., can also be pandas series or numpy array. Defaults to use all features.

            sample_ids (like-like): List of indicators for the samples to use. e.g. [1,3], [True, False, True],
                ['human1', 'human3'], etc..., can also be pandas series or numpy array. Defaults to use all samples.

            norm_name (str): Common name for the normalization used. e.g. log, unit, etc... The default is
                :py:attr:`normalizer`.__str__().

            supervised_attr (string): If not None, the supervised_attr labels are based to the normalizer fit method, rather than None.

            normalize_args (dict)L Dictionary of keyword arguments to be passed to fit_transform method of the
                :py:attr:`normalizer`.
        Returns:
            inplace method.

        Examples:
            >>> from pydataset import data as pydat
            >>> from datasci.core.dataset import DataSet as DS
            >>> from sklearn.preprocessing import StandardScaler  as SS
            >>> df = pydat('iris')
            >>> data = df[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
            >>> metadata = df[['Species']]
            >>> ds = DS(name='Iris', data=data, metadata=metadata)
            >>> normalizer = SS()
            >>> ds.normalize(normalizer=normalizer, norm_name='standard')
        """
        # set defaults
        if norm_name is None:
            norm_name = normalizer.__str__()

        if normalize_args is None:
            normalize_args = {}

        # slice the data set
        ds = self.slice_dataset(feature_ids, sample_ids)
        data = ds.data
        metadata = ds.metadata

        # transform data
        if supervised_attr != None:
            data_trans = normalizer.fit_transform(data.values,
                                                  y=metadata[supervised_attr].values,
                                                  **normalize_args)
        else:
            try:
                data_trans = normalizer.fit_transform(data.values,
                                                      y=None,
                                                      **normalize_args)
            except TypeError:
                data_trans = normalizer.fit_transform(data.values,
                                                      **normalize_args)


        # create dataframe from transformed data
        data_trans = data.__class__(index=data.index, columns=data.columns, data=data_trans)

        # set data
        if data.shape[1] == data_trans.shape[1]:
            self.data.loc[data.index, data.columns] = data_trans
            self.normalization_method = (self.normalization_method + '/' + norm_name).lstrip('/')
        else:
            raise ValueError("Argument \"normalizer\" should not change the number of features.")

    def impute(self, imputer, feature_ids=None, sample_ids=None, impute_name: str = None):
        """
        Imputes the data of the dataset according to an imputer class. Appends the imputation method
        used to :py:attr:`DataSet.imputation_method`.

        Args:
            imputer(object): Class instance which must contain the method fit_transform. The output of
                imputer.fit_transform(:py:attr:`DataSet.data`) must have the same number of columns
                as :py:attr:`DataSet.data`.

            feature_ids (list-like): List of indicators for the features to use. e.g. [1,3], [True, False, True],
                ['gene1', 'gene3'], etc..., can also be pandas series or numpy array. Defaults to use all features.

            sample_ids (like-like): List of indicators for the samples to use. e.g. [1,3], [True, False, True],
                ['human1', 'human3'], etc..., can also be pandas series or numpy array. Defaults to use all samples.

            impute_name (str): Common name for the imputation used. e.g. knn, rf, median, etc .. The default is
                :py:attr:`imputer`.__str__().

        Returns:
            inplace method.

        Examples:
            >>> import pandas as pd
            >>> from datasci.core.dataset import DataSet as DS
            >>> from sklearn.impute import KNNImputer
            >>> data = pd.DataFrame(index=['a', 'b', 'c'],
            ...                     columns= ['x', 'y', 'z'],
            ...                     data=[[1,2,3], [0, 0, 1], [8, 5, 4]])
            >>> ds = DS(name='example', data=data)
            >>> imputer = KNNImputer(missing_values=0, n_neighbors=2)
            >>> ds.impute(imputer=imputer, impute_name='knn')
        """
        # set defaults
        if impute_name is None:
            impute_name = imputer.__str__()

        # slice the data set
        ds = self.slice_dataset(feature_ids, sample_ids)
        data = ds.data

        # transform data
        data_trans = imputer.fit_transform(data.values)

        # create dataframe from transformed data
        data_trans = data.__class__(index=data.index, columns=data.columns, data=data_trans)

        # set data
        if data.shape[1] == data_trans.shape[1]:
            self.data.loc[data.index, data.columns] = data_trans
            self.imputation_method = (self.imputation_method + '/' + impute_name).lstrip('/')
        else:
            raise ValueError("Argument \"imputer\" should not change the number of features.")

    def reformat_metadata(self, convert_dtypes: bool = False):
        """
        This method performs a basic reformatting of metadata including: Replacing double-spaces with a single space,
        Stripping white space from string ends, Removing mixed-case and capitalizing strings. Additionally one can use
        pandas infer_dtypes function to automatically infer the datatypes for each attribute.

        Args:
            convert_dtypes (bool): Flag for whether or not to infer the datatypes for the metadata and vardata.
                Default is false.

        Returns:
            inplace method.

        Examples:
            >>> from pydataset import data as pydat
            >>> from datasci.core.dataset import DataSet as DS
            >>> df = pydat('iris')
            >>> data = df[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
            >>> metadata = df[['Species']]
            >>> ds = DS(name='Iris', data=data, metadata=metadata)
            >>> ds.reformat_metadata(convert_dtypes=True)
        """

        # perform basic cleaning
        self.metadata.replace("  ", " ", inplace=True)  # replace double whitespace
        for column in self.metadata.columns:
            data_series = self.metadata[column]
            if pd.api.types.infer_dtype(data_series) in ["string", "empty", "bytes", "mixed", "mixed-integer"]:
                data_series = data_series.str.lstrip()
                data_series = data_series.str.rstrip()
                data_series = data_series.str.capitalize()
                self.metadata[column] = data_series

        self.vardata.replace("  ", " ", inplace=True)  # replace double whitespace
        for column in self.vardata.columns:
            data_series = self.vardata[column]
            if pd.api.types.infer_dtype(data_series) in ["string", "empty", "bytes", "mixed", "mixed-integer"]:
                data_series = data_series.str.lstrip()
                data_series = data_series.str.rstrip()
                data_series = data_series.str.capitalize()
                self.vardata[column] = data_series

        # convert dtypes
        if convert_dtypes:
            self.metadata = self.metadata.convert_dtypes()
            self.vardata = self.vardata.convert_dtypes()

    def slice_dataset(self, feature_ids=None, sample_ids=None, name=None):
        """
        This method slices a DataSet at the prescribed sample and features ids.

        Args:
            feature_ids (list-like): List of indicators for the features to use. e.g. [1,3], [True, False, True],
                ['gene1', 'gene3'], etc..., can also be pandas series or numpy array. Defaults to use all features.

            sample_ids (like-like): List of indicators for the samples to use. e.g. [1,3], [True, False, True],
                ['human1', 'human3'], etc..., can also be pandas series or numpy array. Defaults to use all samples.

            name (str): Reference name for slice DataSet. Defaults to :py:attr:`DataSet.name` _slice

        Returns:
            DataSet : Slice of DataSet.

        Examples:
            >>> from pydataset import data as pydat
            >>> from datasci.core.dataset import DataSet as DS
            >>> df = pydat('iris')
            >>> data = df[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
            >>> metadata = df[['Species']]
            >>> ds = DS(name='Iris', data=data, metadata=metadata)
            >>> samples = ds.metadata['Species'] == 'setosa'
            >>> ds_setosa = ds.slice_dataset(sample_ids=samples)
        """
        # set defaults
        if feature_ids is None:
            feature_ids = self.data.columns
        if sample_ids is None:
            sample_ids = self.data.index
        if name is None:
            name = self.name + '_slice'

        # sort data, metadata, vardata, and dissimilarity matrix together to be safe
        self.vardata = self.vardata.loc[self.data.columns]
        self.metadata = self.metadata.loc[self.data.index]
        if not(self.dissimilarity_matrix is None):
            self.dissimilarity_matrix = self.dissimilarity_matrix.loc[self.data.index, self.data.index.to_list()]

        # slice data at features
        columns = self.data.columns
        try:
            data = self.data[self.vardata.index[feature_ids]]
        except IndexError:
            try:
                data = self.data[feature_ids]
            except KeyError:
                data = self.data[columns[feature_ids]]

        # slice data at samples
        index = self.data.index
        try:
            data = data.loc[sample_ids]
        except KeyError:
            data = data.loc[index[sample_ids]]

        # slice metadata at samples
        try:
            metadata = self.metadata.loc[sample_ids]
        except KeyError:
            metadata = self.metadata.loc[index[sample_ids]]

        # slice dissimilarity matrix at samples
        if self.dissimilarity_matrix is None:
            dissimilarity_matrix = None
        else:
            try:
                dissimilarity_matrix = self.dissimilarity_matrix.loc[sample_ids, sample_ids.to_list()]
            except KeyError:
                dissimilarity_matrix = self.dissimilarity_matrix.loc[index[sample_ids], index[sample_ids].to_list()]

        # slice vardata at features
        index = self.vardata.index
        try:
            vardata = self.vardata.loc[feature_ids]
        except KeyError:
            vardata = self.vardata.loc[index[feature_ids]]

        # generate slice DataSet
        ds = DataSet(data=deepcopy(data),
                metadata=deepcopy(metadata),
                vardata=vardata,
                dissimilarity_matrix=dissimilarity_matrix,
                name=name,
                normalization_method=self.normalization_method,
                imputation_method=self.imputation_method)

        return ds

    def autosummarize(self, which='metadata', use_dash=False, **kwargs):
        """
        This method gives a human-readable output of summary statistics for the data/metadata/vardata.
        It includes basic statistics such as mean, median, mode, etc. and gives value counts for discrete data
        attributes. When using dash am interactive dashboard will be created. The user will then be able to view
        histograms for each attribute in the metadata along with basic summary statistics. The user can interact
        with the dashboard to adjust the number of bins and attribute.

        Args:
            which (str): String indicating which data to use. Choices are 'data', 'metadata', or 'vardata'. Default
                is 'metadata'.

            use_dash (bool): Flag for indicating whether or not to use dash dashboard. Default is False.

            **kwargs (dict): Passed directly to dash.Dash.app.run_server for configuring host server.
                See dash documentation for further details.

        Returns:
            inplace method.

        Examples:
            >>> from pydataset import data as pydat
            >>> from datasci.core.dataset import DataSet as DS
            >>> df = pydat('iris')
            >>> data = df[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
            >>> metadata = df[['Species']]
            >>> ds = DS(name='Iris', data=data, metadata=metadata)
            >>> ds.autosummarize(use_dash=True, port=8787)
        """
        # set the data
        if which == 'metadata':
            df = self.metadata
        elif which == 'data':
            df = self.data
        elif which == 'vardata':
            df = self.vardata
        else:
            raise ValueError("argument which must be either 'metadata', 'data', or 'vardata'!")

        if use_dash:
            import dash
            import dash_html_components as html
            import dash_core_components as dcc
            from dash.dependencies import Input, Output
            import plotly.graph_objects as go

            # I've always used the same stylesheet but there are options
            external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
            app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

            # the html.Div should open for html code
            # dcc is dash core component sets up the dash element. The first one here is a dropdown menu
            app.layout = html.Div([html.H1(children=self.name + ' ' + which.capitalize() + " Auto-Summary"),
                                   html.Div(children='''Choose an attribute below.''', style={'padding': 10}),

                                   html.Div([
                                       dcc.Dropdown(
                                           id='attribute',
                                           options=[{'label': attr, 'value': attr} for attr in df.columns],
                                           value=df.columns[0],
                                           clearable=False)]),

                                   html.Div(children='''Choose the number of bins (0 = Auto).''',
                                            style={'padding': 10}),

                                   html.Div([
                                       dcc.Input(id="bins",
                                                 type="number",
                                                 placeholder="Debounce False",
                                                 value=0,
                                                 min=0)]),

                                   html.Div([
                                       dcc.Graph(id='freq-figure')], className="row"),

                                   html.Hr(),
                                   ])

            # multiple inputs order matters in defining function below.
            @app.callback(
                Output('freq-figure', 'figure'),
                [Input('attribute', 'value'), Input('bins', 'value')])
            # make a histogram figure
            def freq_fig(attr, bins):
                bins = int(bins)
                series = df[attr]
                nan_count = np.sum(series.isna())
                fig = go.Figure()
                if pd.api.types.infer_dtype(series) in ["floating", "integer"]:
                    if bins == 0:
                        bins = 'auto'
                    y, xx = np.histogram(series[-series.isna()], bins=bins)
                    x = (xx[1:] + xx[:-1]) / 2
                    # xx = np.hstack((x, xx))
                    # xx = np.sort(xx)
                    xx = x
                    fig.add_trace(go.Bar(x=x, y=y))

                    fig.add_annotation(
                        x=0,
                        y=-.175,
                        xref="paper",
                        yref="paper",
                        text="Max = " + str(series.max()),
                        font=dict(
                            family="Courier New, monospace",
                            size=18,
                        ),
                        showarrow=False,
                    )

                    fig.add_annotation(
                        x=0,
                        y=-.275,
                        xref="paper",
                        yref="paper",
                        text="Min = " + str(series.min()),
                        font=dict(
                            family="Courier New, monospace",
                            size=18,
                        ),
                        showarrow=False,
                    )

                    fig.add_annotation(
                        x=1,
                        y=-.175,
                        xref="paper",
                        yref="paper",
                        text="Mean = " + str(series.mean()),
                        font=dict(
                            family="Courier New, monospace",
                            size=18,
                        ),
                        showarrow=False,
                    )

                    fig.add_annotation(
                        x=1,
                        y=-.275,
                        xref="paper",
                        yref="paper",
                        text="Median = " + str(series.median()),
                        font=dict(
                            family="Courier New, monospace",
                            size=18,
                        ),
                        showarrow=False,
                    )

                else:
                    freq_series = series.value_counts()
                    x = freq_series.index
                    x = pd.Series(x)
                    y = freq_series.values
                    fig.add_trace(go.Bar(x=x, y=y))

                    fig.add_annotation(
                        x=1,
                        y=-.175,
                        xref="paper",
                        yref="paper",
                        text="Mode = " + '/'.join(series.mode().tolist()),
                        font=dict(
                            family="Courier New, monospace",
                            size=18,
                        ),
                        showarrow=False,
                    )
                    xx = x

                fig.add_annotation(
                    x=1,
                    y=1.175,
                    xref="paper",
                    yref="paper",
                    text="Missing = " + str(nan_count),
                    font=dict(
                        family="Courier New, monospace",
                        size=18,
                    ),
                    showarrow=False,
                )

                fig.update_layout(title_text="Histogram of " + attr + " Attribute",
                                  xaxis_title=attr,
                                  yaxis_title="Frequency",
                                  title_x=0.5,
                                  xaxis=dict(
                                      tickmode='array',
                                      tickvals=xx),
                                  )

                return fig

            app.run_server(**kwargs)

        else:
            for column in df.columns:
                series = df[column]
                header = "Summary statistics for " + column + " attribute:"
                print('-' * len(header))
                print(header)
                print('-' * len(header))
                stats = series.describe()
                stats.at['missing'] = np.sum(series.isna())
                print(stats)
                print('-' * len(header))
                if pd.api.types.infer_dtype(series) in ["floating", "integer"]:
                    print('\n')
                else:
                    print('Value counts:')
                    print('-' * len(header))
                    print(series.value_counts())
                    print('-' * len(header))
                    print('\n')

    def save(self, file_path: str = None):
        """
        This method saves an instance of a DataSet class in pickle format. If no path is given the instance will save
        as :py:attr:`DataSet.path`/:py:attr:`DataSet.name`.ds where the spaces in :py:attr:`DataSet.name` are replaced
        with underscores.

        Args:
            file_path (str): Path of the file to save the instance of DataSet to. Default is None.

        Returns:
            inplace method.

        Examples:
            >>> from pydataset import data as pydat
            >>> from datasci.core.dataset import DataSet as DS
            >>> df = pydat('iris')
            >>> data = df[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
            >>> metadata = df[['Species']]
            >>> ds = DS(name='Iris', data=data, metadata=metadata)
            >>> ds.save()
        """
        # check path
        if file_path is None:
            file_path = self.path + '/' + self.name.replace(" ", "_") + '.ds'

        # pickle class instance
        with open(file_path, 'wb') as f:
            pickle.dump(self, file=f)

    def print_description(self, line_width: int = 50):
        """
        This method prints the description of the dataset in a human-readable format.

        Args:
            line_width (int): Number of characters per line to print for description.

        Returns:
            inplace method.
        """

        import textwrap
        print(textwrap.fill(self._description, line_width, break_long_words=False, replace_whitespace=False))

    def classify(self, classifier,
                 attr: str,
                 classifier_name=None,
                 fit_args: dict = {},
                 predict_args: dict = {},
                 feature_ids=None,
                 sample_ids=None,
                 partitioner=None,
                 partitioner_name=None,
                 scorer=None,
                 scorer_args: dict = {},
                 scorer_name=None,
                 split_handle: str = 'split',
                 fit_handle: str = 'fit',
                 predict_handle: str = 'predict',
                 f_weights_handle: str = None,
                 s_weights_handle: str = None,
                 append_to_meta: bool = True,
                 inplace: bool = False,
                 f_rnk_func=None,
                 s_rnk_func=None,
                 training_transform=None,
                 experiment_name=None,
                 verbose: bool = True,
                 **kwargs):
        """
        This method runs a classification experiment. The user provides a classifier, a class to partition the data
        into train/test partitions, and a scoring method. The experiment returns the fit classifiers across the
        train/test partitions, the train/test scores. Additionally, the experiment will either return, or append
        depending on the ``append`` flag, the prediction labels, the training and testing labels, feature/sample
        weights associated to the fit classifiers, and their associated rankings.

        Args:
            classifier (object): Classifier to run the classification experiment with; must have the sklearn equivalent
                of a ``fit`` and ``predict`` method.

            fit_args (dict): Keyword arguments passed to the classifiers fit method.

            predict_args (dict): Keyword arguments passed to the classifers predict method

            attr (string): Name of metadata attribute to classify on.

            classifier_name (string): Common name of classifier to be used for identification. Default is
                ``classifier.__str__()``.

            feature_ids (list-like): List of indicators for the features to use. e.g. [1,3], [True, False, True],
                ['gene1', 'gene3'], etc..., can also be pandas series or numpy array. Defaults to use all features.

            sample_ids (like-like): List of indicators for the samples to use. e.g. [1,3], [True, False, True],
                ['human1', 'human3'], etc..., can also be pandas series or numpy array. Defaults to use all samples.

            partitioner (object): Option 1.) Class-instance which partitions samples in batches of training and test
                split. This instance must have the sklearn equivalent of a split method. The split method returns a
                list of train-test partitions; one for each fold in the experiment. See sklearn.model_selection.KFold
                for an example partitioner. Option 2.) Tuple of training and test ids. The default is None; resulting
                in using all of the samples to train the classifier with no test samples.

            partitioner_name (string): Common name for the ``partitioner`` to be used for identification. Default is
                ``partitioner.__str__()``.

            scorer (object): Function which scores the prediction labels on training and test partitions. This function
                should accept two arguments: truth labels and prediction labels. This function should output a score
                between 0 and 1 which can be thought of as an accuracy measure. See
                sklearn.metrics.balanced_accuracy_score for an example.

            scorer_args (dict): Keyword argumunts passed to the scoring function used.

            scorer_name (string): Common name of scorer to used for identification. Default is ``scorer.__str__()``.

            split_handle (string): Name of ``split`` method used by ``partitioner``. Default is "split".

            fit_handle (string): Name of ``fit`` method used by ``classifier``. Default is "fit".

            predict_handle (string): Name of ``predict`` method used by ``classifier``. Default is ``predict``.

            f_weights_handle (string): Name of ``classifier`` attribute containing feature weights. Default is None.

            s_weights_handle (string): Name of ``classifier`` attribute containing sample weights. Default is None.

            append_to_meta (bool): If ``True``, the classification results will be appended to
            :py:attr:`DataSet.metadata` and :py:attr:`DataSet.vardata`. Default is ``False``.

            inplace (bool): If True the classification results will be stored to :py:attr:`DataSet.experiments`.
                If ``False`` the classification results will be returned to the user. Default is ``False``

            f_rnk_func (object): Function to be applied to feature weights for feature ranking. Default is None, and the
                features will be ranked from greatest to least importance. e.g. rank = 1 most important.

            s_rnk_func (object): Function to be applied to sample weights for sample ranking. Default is None, and the
                samples will be ranked in from least to greatest.

            training_transform (object): Transformer to be fit on training partitions and applied to both training
                and test data. For example fit a StandardScalar transform to the training data and apply the learned
                affine transform to the training and test data. This is useful for on the fly normalization.
                The default is None.

            experiment_name (string): Common name of experiment to use when ``inplace=True`` and storing results into
                :py:attr:`DataSet.experiments`. Default is ``attr`` + ``classifier_name`` + ``partitioner_name`` + ``scorer_name``.

            verbose (boolean): If True, logs will be printed to console. Default = True
        Returns:
            dict : (classifiers) - Contains the fit classifiers. (scores) - Contains the training and test scores
                provided by ``scorer`` across partitions given by ``partitioner``. (prediction_results) - Contains the
                prediction labels and train/test labels across each fold generated by ``partitioner``. (f_weights) -
                Contains the feature weights and rankings in the classification experiment for each fold generated by
                ``partitioner``. (s_weights) - Contains the sample weights and rankings in the classification experiment
                for each fold generated by ``partitioner``.

        Examples:
            >>> # imports
            >>> import datasci.core.dataset as dataset
            >>> from datasci.sparse.classifiers.svm import SSVMClassifier as SSVM
            >>> from calcom.solvers import LPPrimalDualPy
            >>> from sklearn.model_selection import KFold
            >>> from sklearn.metrics import balanced_accuracy_score as bsr
            ...
            >>> # load dataset
            >>> ds = dataset.load_dataset('./test_data/GSE161731_tmm_log2.ds')
            ...
            >>> # setup classification experiment
            >>> ssvm = SSVM(solver=LPPrimalDualPy, use_cuda=True)
            >>> kfold = KFold(n_splits=5, shuffle=True, random_state=0)
            >>> covid_healthy = ds.metadata['cohort'].isin(['COVID-19', 'healthy'])
            ...
            >>> # run classification
            >>> ds.classify(classifier=ssvm,
            ...             classifier_name='SSVM',
            ...             attr='cohort',
            ...             sample_ids=covid_healthy,
            ...             partitioner=kfold,
            ...             partitioner_name='5-fold',
            ...             scorer=bsr,
            ...             scorer_name='bsr',
            ...             f_weights_handle='weights_',
            ...             append_to_meta=True,
            ...             inplace=True,
            ...             experiment_name='covid_vs_healthy_SSVM_5-fold',
            ...             f_rnk_func=np.abs)
            ...
            >>> # share the results
            >>> ds.save('./test_data/GSE161731_ssvm_results.ds')
        """
        # set defaults
        if training_transform is None:
            training_transform = FunctionTransformer(lambda x: x)

        if classifier_name is None:
            classifier_name = classifier.__str__().split('(')[0]

        if partitioner_name is None:
            partitioner_name = partitioner.__str__().split('(')[0]

        if scorer_name is None:
            scorer_name = scorer.__str__().split('(')[0]

        method_name = attr + "_" + partitioner_name + "_" + classifier_name

        # slice dataframe
        ds = self.slice_dataset(feature_ids, sample_ids)
        sample_ids = ds.data.index
        feature_ids = ds.vardata.index
        X = ds.data.values
        y = ds.metadata[attr].infer_objects().values

        # TODO: Fix sample_ids for passing to numpy array
        if partitioner is None:
            splits = [(np.arange(0, len(sample_ids)), None)]
        elif type(partitioner) == tuple:
            train_ids = self.slice_dataset(sample_ids=partitioner[0]).data.index
            test_ids = self.slice_dataset(sample_ids=partitioner[1]).data.index
            train_ids = np.where(ds.data.index.isin(train_ids))[0]
            test_ids = np.where(ds.data.index.isin(test_ids))[0]
            splits = [(train_ids, test_ids)]
        else:
            split = eval("partitioner" + "." + split_handle)
            group = kwargs.get('group', None)

            if group is None:
                groups = ds.data.index
            else:
                groups = ds.metadata[group].values
                
            try:
                splits = split(X, y, groups=groups)
            except (AttributeError, TypeError):
                splits = split(X, y)
            try:
                _ = (e for e in splits)
            except TypeError:
                splits = [splits]

        # set sample index
        sample_index = self.metadata.loc[sample_ids].index

        # set returns
        predict_results = pd.DataFrame(index=sample_ids)
        f_weight_results = pd.DataFrame(index=feature_ids)
        s_weight_results = pd.DataFrame(index=sample_ids)
        scores = pd.DataFrame(index=['Train', 'Test'])
        classifiers = pd.Series()

        # set methods
        fit = eval("classifier" + "." + fit_handle)
        predict = eval("classifier" + "." + predict_handle)

        # loop over splits
        try:
            n_splits = partitioner.get_n_splits(groups=groups)
        except AttributeError:
            n_splits = len(splits)
        for i, (train_index, test_index) in enumerate(splits):
            X_train = training_transform.fit_transform(X[train_index, :])
            if not (test_index is None):
                X_test = training_transform.transform(X[test_index, :])
            y_train = y[train_index]
            if not (test_index is None):
                y_true = y[test_index]

            # fit the classifier (going meta y'all)
            fit(X_train, y_train, **fit_args)
            classifiers[method_name + '_classifier_' + str(i)] = classifier

            # get predict labels
            y_pred_train = predict(X_train, **predict_args)
            if not (test_index is None):
                y_pred_test = predict(X_test, **predict_args)

            # get scores
            title = r"%s, Split %d of %d, Scores: " % (classifier_name, i+1, n_splits)
            print_str = title + '\n'
            print_str += "="*len(title) + '\n'
            score_name = method_name + "_" + scorer_name + "_" + str(i)
            train_score = scorer(y_train, y_pred_train, **scorer_args)
            if pd.api.types.infer_dtype(train_score) == 'floating':
                print_str += r"Training %s: %.2f%%" % (scorer_name, train_score*100) + '\n'
            else:
                print_str += r"Training %s:" % (scorer_name,) + '\n'
                print_str += str(train_score) + '\n'
            if not (test_index is None):
                print_str += "-"*len(title) + '\n'
                test_score = scorer(y_true, y_pred_test, **scorer_args)
                if pd.api.types.infer_dtype(test_score) == 'floating':
                    print_str += r"Test %s: %.2f%%" % (scorer_name, test_score * 100) + '\n'
                else:
                    print_str += r"Test %s:" % (scorer_name,) + '\n'
                    print_str += str(test_score) + '\n'
                scores[score_name] = pd.Series(index=['Train', 'Test'], data=[train_score, test_score])
            else:
                scores[score_name] = pd.Series(index=['Train', 'Test'], data=[train_score, pd.NA])
            print_str += "\n"
            if verbose:
                print(print_str)
            # append prediction results
            labels_name = method_name + "_labels_" + str(i)
            split_labels_name = method_name + "_split_labels_" + str(i)
            predict_results[labels_name] = ''
            predict_results.loc[sample_index.take(train_index), labels_name] = y_pred_train
            if not (test_index is None):
                predict_results.loc[sample_index.take(test_index), labels_name] = y_pred_test
            predict_results[split_labels_name] = ''
            predict_results.loc[sample_index.take(train_index), split_labels_name] = 'Train'
            if not (test_index is None):
                predict_results.loc[sample_index.take(test_index), split_labels_name] = 'Test'

            # append feature results
            if not (f_weights_handle is None):
                f_weights_name = method_name + "_f_weights_" + str(i)
                f_weights = eval("classifier" + "." + f_weights_handle).reshape(-1)
                f_weight_results[f_weights_name] = np.nan
                f_weight_results.loc[feature_ids, f_weights_name] = pd.Series(index=feature_ids, data=f_weights)
                if not (f_rnk_func is None):
                    f_rnk_name = method_name + "_f_rank_" + str(i)
                    weights = f_weight_results.loc[feature_ids, f_weights_name]
                    f_weight_results[f_rnk_name] = np.nan
                    f_weight_results.loc[feature_ids, f_rnk_name] = (-f_rnk_func(weights)).argsort()
                    f_weight_results[f_rnk_name] = f_weight_results[f_rnk_name].astype('Int64')
                else:
                    f_rnk_name = method_name + "_f_rank_" + str(i)
                    weights = f_weight_results.loc[feature_ids, f_weights_name]
                    f_weight_results[f_rnk_name] = np.nan
                    f_weight_results.loc[feature_ids, f_rnk_name] = (-np.array(weights)).argsort()
                    f_weight_results[f_rnk_name] = f_weight_results[f_rnk_name].astype('Int64')

            # append sample results
            if not (s_weights_handle is None):
                s_weights_name = method_name + "_s_weights_" + str(i)
                s_weights = eval("classifier" + "." + s_weights_handle)
                s_weight_results[s_weights_name] = np.nan
                s_weight_results.loc[sample_index.take(train_index), s_weights_name] = s_weights
                if not (s_rnk_func is None):
                    s_rnk_name = method_name + "_s_rank_" + str(i)
                    weights = s_weight_results.loc[sample_index.take(train_index), s_weights_name]
                    s_weight_results[s_rnk_name] = np.nan
                    s_weight_results.loc[sample_index.take(train_index), s_rnk_name] = (-s_rnk_func(weights)).argsort()
                    s_weight_results[s_rnk_name] = s_weight_results[s_rnk_name].astype('Int64')
                else:
                    s_rnk_name = method_name + "_s_rank_" + str(i)
                    weights = s_weight_results.loc[sample_index.take(train_index), s_weights_name]
                    s_weight_results[s_rnk_name] = np.nan
                    s_weight_results.loc[sample_index.take(train_index), s_rnk_name] = (-np.array(weights)).argsort()

        # print means and standard deviations
        print_str = ''
        if pd.api.types.infer_dtype(train_score) == 'floating':
            title = r"%s, Summary, Scores: " % (classifier_name,)
            print_str += title + '\n'
            print_str += "=" * len(title) + '\n'
            mean_train_score = scores.loc['Train'].mean()
            std_train_score = scores.loc['Train'].std()
            min_train_score = scores.loc['Train'].min()
            max_train_score = scores.loc['Train'].max()
            print_str += r"Training %s: %.2f%% +/- %.2f%%" % (scorer_name, mean_train_score * 100, std_train_score * 100) + '\n'
            print_str += r"Max. Training %s: %.2f%%" % (scorer_name, max_train_score * 100) + '\n'
            print_str += r"Min. Training %s: %.2f%%" % (scorer_name, min_train_score * 100) + '\n'
        if not (test_index is None):
            if pd.api.types.infer_dtype(test_score) == 'floating':
                print_str +="-"*len(title) + '\n'
                mean_test_score = scores.loc['Test'].mean()
                std_test_score = scores.loc['Test'].std()
                min_test_score = scores.loc['Test'].min()
                max_test_score = scores.loc['Test'].max()
                print_str +=r"Test %s: %.2f%% +/- %.2f%%" % (scorer_name, mean_test_score * 100, std_test_score * 100) + '\n'
                print_str +=r"Max. Test %s: %.2f%%" % (scorer_name, max_test_score * 100) + '\n'
                print_str +=r"Min. Test %s: %.2f%%" % (scorer_name, min_test_score * 100) + '\n'

        if verbose:
            print(print_str)
        if append_to_meta:
            self.metadata = pd.concat([self.metadata, predict_results], axis=1)
            self.vardata = pd.concat([self.vardata, f_weight_results], axis=1)
            self.metadata = pd.concat([self.metadata, s_weight_results], axis=1)

        results = {'classifiers': classifiers,
                   'scores': scores,
                   'prediction_results': predict_results,
                   'f_weights': f_weight_results,
                   's_weights': s_weight_results}

        if inplace:
            if experiment_name is None:
                experiment_name = attr + '_' + classifier_name + '_' + partitioner_name + '_' + scorer_name
            self.experiments[experiment_name] = results
        else:
            return results

    def feature_select(self, selector,
                 attr: str,
                 cross_attr: str = None,
                 selector_name=None,
                 feature_ids=None,
                 sample_ids=None,
                 fit_handle: str = 'fit',
                 f_results_handle: str = 'results_',
                 append_to_meta: bool = True,
                 inplace: bool = False,
                 training_transform=None,
                 experiment_name=None,
                 ):
        """
        This method runs a feature selection experiment. The user provides a feature selector and a ranking function.
        The experiment returns, or appends depending on the ``append`` flag, the feature weights, and their
        associated rankings.

        Args:
            selector (object): Feature selector to run the feature selection experiment with; must have the
                sklearn equivalent of a ``fit`` method.

            attr (string): Name of metadata attribute to feature select on.

            cross_attr (string): (Optional) Name of metadata cross attribute to feature select on. The ``fit`` method
                of the ``selector`` must accept these cross labels.

            selector_name (string): Common name of feature selector to be used for identification. Default is
                ``selector.__str__()``.

            feature_ids (list-like): List of indicators for the features to use. e.g. [1,3], [True, False, True],
                ['gene1', 'gene3'], etc..., can also be pandas series or numpy array. Defaults to use all features.

            sample_ids (like-like): List of indicators for the samples to use. e.g. [1,3], [True, False, True],
                ['human1', 'human3'], etc..., can also be pandas series or numpy array. Defaults to use all samples.

            fit_handle (string): Name of ``fit`` method used by ``selector``. Default is "fit".

            f_results_handle (string): Name of ``selector`` attribute containing feature results e.g. weights, ranks,
                etc.The attribute should be array-like with rows corresponding to the features. Default is "results_".

            append_to_meta (bool): If ``True``, the feature selection results will be appended to
                :py:attr:`DataSet.metadata` and :py:attr:`DataSet.vardata`. Default is ``False``.

            inplace (bool): If True the feature selection results will be stored to :py:attr:`DataSet.experiments`.
                If ``False`` the feature selection results will be returned to the user. Default is ``False``

            training_transform (object): Transformer to be fit on training partitions and applied to both training
                and test data. For example fit a StandardScalar transform to the training data and apply the learned
                affine transform to the training and test data. This is useful for on the fly normalization.
                The default is None.

            experiment_name (string): Common name of experiment to use when ``inplace=True`` and storing results into
                :py:attr:`DataSet.experiments`. Default is ``attr``+ ``selector_name``.
        Returns:
            dict : (selector) - Contains the fit feature selector. (f_weights) - Contains the feature weights and
            rankings in the feature selection experiment.

        Examples:
            >>> # imports
            >>> import numpy as np
            >>> import datasci.core.dataset as dataset
            >>> from datasci.sparse.classifiers.svm import SSVMClassifier as SSVM
            >>> from calcom.solvers import LPPrimalDualPy
            >>> from datasci.sparse.feature_selection.kffs import KFFS
            ...
            >>> # load dataset
            >>> ds = dataset.load_dataset('./test_data/GSE161731_tmm_log2.ds')
            ...
            >>> # setup classification experiment
            >>> ssvm = SSVM(solver=LPPrimalDualPy, use_cuda=True)
            >>> kffs = KFFS(k=5,
            ...             n=5,
            ...             classifier=ssvm,
            ...             f_weights_handle = 'weights_',
            ...             f_rnk_func=np.abs,
            ...             random_state=0)
            ...
            >>> covid_healthy = ds.metadata['cohort'].isin(['COVID-19', 'healthy'])
            ...
            >>> # run classification
            >>> ds.feature_select(selector=kffs,
            ...             selector_name='kFFS',
            ...             attr='cohort',
            ...             sample_ids=covid_healthy,
            ...             f_results_handle='results_',
            ...             append_to_meta=True,
            ...             inplace=True,
            ...             experiment_name='covid_vs_healthy_SSVM_kFFS')
        """
        # set defaults
        if training_transform is None:
            training_transform = FunctionTransformer(lambda x: x)

        if selector_name is None:
            selector_name = selector.__str__().split('(')[0]

        method_name = attr + "_" + selector_name

        # slice dataframe
        ds = self.slice_dataset(feature_ids, sample_ids)
        sample_ids = ds.data.index
        feature_ids = ds.vardata.index
        X = ds.data.values
        y = ds.metadata[attr].infer_objects().values

        # set returns
        f_weight_results = pd.DataFrame(index=self.vardata.index)

        # set methods
        fit = eval("selector" + "." + fit_handle)

        # fit the feature selector (going meta y'all)
        if cross_attr is None:
            fit(training_transform.fit_transform(X), y)
        else:
            groups = ds.metadata[cross_attr].values
            fit(X, y, groups)

        # append feature results
        f_results = pd.DataFrame(eval("selector" + "." + f_results_handle))	
		
        # set returns	
        f_weight_results = pd.DataFrame(index=self.vardata.index, columns=f_results.columns)	          
        f_weight_results.loc[feature_ids] = pd.DataFrame(data=f_results.values, index=feature_ids, columns=f_results.columns)	

        results = {'selector': selector,
                   'f_results': f_weight_results}

        if append_to_meta:
            self.vardata = pd.concat([self.vardata, f_weight_results], axis=1)

        if inplace:
            if experiment_name is None:
                experiment_name = attr + '_' + selector_name
            self.experiments[experiment_name] = results
        else:
            return results


    def generate_attr_from_queries(self, 	
	                                attrname:str, 	
	                                queries: dict, 	
	                                attr_exist_mode: str = 'err',	
	                                which: str='metadata'):	
		
	        """	
	        This function creates or updates an attribute in the metadata or vardata. New values for the attribute	
	        are provided by the queries, which is a dictionary. For each value in the queries dictionary, indices are 	
	        extracted using the query method on the dataframe and the key is used as new value at these indices. Any index 	
	        which is not covered by any of the query is set to pandas.NA	
	        Args:	
	            attrname (str): Name of the new attribute	
	            queries (dict): key: label for the new attribute at the filtered indices, value: query string to filter the indices	
	            attr_exist_mode (str) : 'err' : raises an Exception if the attribute already exists in the dataframe	
	                                    'overwrite' : overwrites the previous values with new values	
	                                    'append' : updates and appends "_x"  the attribute name, where x is an integer based on existing attributes names.	
	                                    Ex. if 'response', 'response_new', 'response_1' is already present, the new name for the attribute will be 'response_2'	
	            which (str): String indicating which data to use. Choices are 'metadata' or 'vardata'. Default	
	                is 'metadata'.	
	        Returns:	
	            inplace method	
	        Examples:	
	                >>> q_res = "Tissue=='Liver' and response_new=='resistant' and partition in ['training', 'validation']"	
	                >>> q_tol = "Tissue=='Liver' and response_new=='tolerant' and partition in ['training', 'validation']	
	                >>> attribute_name = 'Response'	
	                >>> qs = {'Resistant' : q_res, 'Tolerant': q_tol}	
	                >>> ds.generate_attr_from_queries(attribute_name, qs, attr_exist_mode='append')	
	        """	
	        if which == 'metadata':	
	            df = self.metadata	
	        elif which == 'vardata':	
	            df = self.vardata	
		
	        attr_exists = False	
	        # check if attr exists	
	        if attrname in df.columns:	
	            attr_exists = True	
		
	        if attr_exists:	
	            if attr_exist_mode == 'err':	
	                raise Exception("Attribute '%s' already exists. Please provide a different attribute name or change attr_exist_mode to 'append' or 'overwrite'." % attrname)	
		
	            elif attr_exist_mode == 'append':	
	                existing_column_names = df.filter(regex=attrname).columns	
	                splits = existing_column_names.str.split('_')	
	                reps = []	
		
	                #find the largest rep number	
	                for split in splits:	
	                    try:	
	                        #convert the last entry to int	
	                        #may throw an exception when converting to int	
	                        #if any column name has an underscore, 	
	                        #ex. if column name is response_new, it's split will be ['response' 'new'] 	
	                        rep = int(split[-1])	
	                        print(rep)	
	                    except:	
	                        continue	
	                    reps.append(rep)	
	                reps = np.sort(np.array(reps))	
	                	
	                attrname = attrname + '_%d'%(reps[-1]+1)	
	        	
	            elif attr_exist_mode != 'overwrite':	
	                raise Exception("Incorrect value passed for attr_exist_mode. Allowed values are 'err', 'append' and 'overwrite'")	
		
	        df[attrname] = pd.NA	
	        for key, values in queries.items():	
	            index = df.query(values).index	
	            df.loc[index, attrname] = key	
	
    # TODO: Add conversion to cdd method.

    # class methods

# functions
def load_dataset(file_path: str):
    """
    This function loads and returns an instance of a DataSet class in pickle format.

    Args:
        file_path (str): Path of the file to load the instance of DataSet from.

    Returns:
        DataSet : Class instance encoded by pickle binary file_path.

    Examples:
            >>> ds = load_dataset(file_path=os.path.join(os.environ["DATASCI_PATH"], "test_data/Iris/Data/iris.ds"))
    """
    # open file and unpickle
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def from_ccd(file_path: str, name: str = None, index_col: str = '_id'):
    """
    This function loads a Calcom Dataset object and returns an instance of a DataSet class.
    Args:
        file_path (str): Path of the CCDataSet file to load.
        name (str): Reference name for the dataset. Default is the name of the ccd file (without extension).
        index_col (str): attribute name from the ccd file to use as index for data and metadata dataframes (must contain unique values).
    Returns:
        DataSet : Class instance of Dataset.
    Examples:
            >>> ds = from_ccd(file_path='/path/to/ccd_file.h5')
    """
    import calcom
    # load ccdataset
    ccd = calcom.io.CCDataSet(file_path)

    # get index column values
    try:
        index_vals = ccd.get_attrs(index_col)
    except:
        print('Attribute %s not found, using _id as index' % index_col)
        index_vals = ccd.get_attrs('_id')

    # check index column values are unique
    assert np.unique(index_vals).shape[0] == index_vals.shape[
        0], '%s cannot be used as index, it contains duplicate values for samples.' % index_col

    # if filename not provided
    if name is None:
        # use ccd file name without extensions
        name = os.path.splitext(os.path.basename(file_path))[0]

    description = ccd._about_str

    path = os.path.dirname(ccd.fname)

    data = ccd.generate_data_matrix()
    data_df = df = pd.DataFrame(data, columns=ccd.variable_names)
    data_df.index = index_vals

    metadata = None
    for attr in ccd.attrs:
        if metadata is None:
            metadata = ccd.get_attrs(attr).reshape(-1, 1)
        else:
            metadata = np.hstack((metadata, ccd.get_attrs(attr).reshape(-1, 1)))
    metadata_df = pd.DataFrame(metadata, columns=ccd.attrs)
    metadata_df.index = index_vals

    # create and return DS object
    ds = DataSet(name=name, description=description, path=path, data=data_df, metadata=metadata_df)
    return ds



#def load_geo(**kwargs):
    # imports
    #import GEOparse

    # load geo file
    #gse = GEOparse.get_GEO(**kwargs)

    # grab metadata
    #metadata = gse.phenotype_data

    # grab data
    #data = pd.DataFrame(index=metadata.index)
    #for sample in gse.gsms:
        #data.loc[sample] = gse.gsms[sample].table['VALUE']

    # get name
    #name = gse.name

# TODO: Add

if __name__ == "__main__":
    from datasci.core.dataset import DataSet as DS
    import pandas as pd

    # load data
    data = pd.read_csv('/hdd/Zoetis/Data/GSE150706_data.csv', index_col=0)

    # load metadata
    metadata = pd.read_csv('/hdd/Zoetis/Data/GSE150706_metadata.csv', index_col=0)

    # description
    description = 'Longitudinal blood transcriptomic analysis to identify molecular regulatory patterns of Bovine Respiratory Disease in beef cattle. We profiled blood transcriptomics of 24 beef steers at three important stages (Entry: on arrival at the feedlot; Pulled: when sickness is identified; and Close-out: recovered, healthy cattle at shipping to slaughter) to reveal the key biological functions and regulatory factors of BRD and identify gene markers of BRD for early diagnosis and potentially use in selection.'

    # name
    name = 'GSE150706_raw'

    # set dataset
    ds = DS(data=data, metadata=metadata, name=name, description=description)

    # save dataset
    ds.save('/hdd/Zoetis/Data/GSE150706_raw.ds')
