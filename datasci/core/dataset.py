"""
This module contains the main DataSet class used for all preprocessing, visualization, and classification.
"""

# imports
import os
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
from numpy.core import ndarray
from pandas.core.frame import DataFrame
from pandas.core.frame import Series
from datasci.core.helper import scatter_pyplot
from datasci.core.helper import scatter_plotly

# classes
class DataSet:
    """
    Primary base class for storing data and metadata for a generic dataset. Contains methods for quick data
    pre-processing, visualization, and classification.

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

        normalization_method (str): Label indicating the normalization used on the data. Future normalization will
            append as normalization_1/normalization_2/.../normalization_n indicating the sequence of normalizations
            used on the data. Default is the empty string.

        imputation_method (str): Label indicating the imputation used on the data. Default is the empty string.

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
                 vardata: DataFrame = pd.DataFrame(),
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

        # private attributes

        # restrict metadata to data
        meta_index = self.metadata.index.intersection(self.data.index)
        missing_data_index = self.data.index.drop(meta_index)
        missing_data = pd.DataFrame(index=missing_data_index)
        self.metadata = self.metadata.loc[meta_index].append(missing_data)

        # sort data and metadata to be in same order
        self.data.sort_index(inplace=True)
        self.metadata.sort_index(inplace=True)

        # Assign vardata
        if vardata is None:
            self.vardata = pd.DataFrame(index=self.data.columns)
        else:
            # restrict metadata to data
            var_index = vardata.index.intersection(self.data.transpose().index)
            missing_data_index = self.data.transpose().index.drop(var_index)
            missing_data = pd.DataFrame(index=missing_data_index)
            self.vardata = vardata.loc[var_index].append(missing_data)

    def visualize(self, embedding,
                  attr: str,
                  cross_attr: str = None,
                  feature_ids=None,
                  sample_ids=None,
                  backend: str = 'pyplot',
                  viz_name: str = None,
                  save: bool = False,
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

            backend (str): Plotting backend to use. Can be either ``pyplot`` or ``plotly``. The default is ``pyplot``.

            viz_name (str): Common name for the embedding used. e.g. MDS, PCA, UMAP, etc... The default is
                :py:attr:`embedding`.__str__().

            save (bool): Flag indicating to save the file. The file will save to self.path with the file name
                :py:attr:`DataSet.name` _ :py:attr:`viz_name` _ :py:attr:`attrname`.png for ``pyplot`` and
                :py:attr:`DataSet.name` _ :py:attr:`viz_name` _ :py:attr:`attrname`.html for ``plotly``

            **kwargs (dict): Keyword arguments passed directly to :py:func:`helper.scatter_pyplot` when using the
                backend ``pyplot`` and :py:func:`helper.scatter_plotly` when using the backend ``plotly``, for
                indicating plot properties.

        Returns:
            inplace method.

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
            viz_name = embedding.__str__()

        # slice the data set
        ds = self.slice_dataset(feature_ids, sample_ids)
        data = ds.data
        metadata = ds.metadata

        # transform data
        data_trans = embedding.fit_transform(data.values)
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
        metadata = deepcopy(metadata.fillna('nan'))

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
            save_name = self.path + self.name + '_' + viz_name + '_' + str(imputation_method) + '_' + str(
                normalization_method) + '_' + str(attr) + '_' + str(dim)
        else:
            title = 'Visualization of data set ' + self.name + ' using\n' \
                    + viz_name + ' with labels given by ' + attr + ' and ' + cross_attr
            save_name = self.path + self.name + '_' + viz_name + '_' + str(imputation_method) + '_' + str(
                normalization_method) + '_' + str(attr) + '_' + str(cross_attr) + '_' + str(dim)

        if not save:
            save_name = None

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

    def normalize(self, normalizer, feature_ids=None, sample_ids=None, norm_name: str = None):
        """
        Normalizes the data of the dataset according to a normalizer class. Appeneds the normalization method
        used to :py:attr:`DataSet.normalization_method`.

        Args:
            normalizer: Class instance which must contain the method fit_transform. The output of
                normalizer.fit_transform(:py:attr:`DataSet.data`) must have the same number of columns
                as :py:attr:`DataSet.data`.

            feature_ids (list-like): List of indicators for the features to use. e.g. [1,3], [True, False, True],
                ['gene1', 'gene3'], etc..., can also be pandas series or numpy array. Defaults to use all features.

            sample_ids (like-like): List of indicators for the samples to use. e.g. [1,3], [True, False, True],
                ['human1', 'human3'], etc..., can also be pandas series or numpy array. Defaults to use all samples.

            norm_name (str): Common name for the normalization used. e.g. log, unit, etc... The default is
                :py:attr:`normalizer`.__str__().

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

        # slice the data set
        ds = self.slice_dataset(feature_ids, sample_ids)
        data = ds.data

        # transform data
        data_trans = normalizer.fit_transform(data.values)

        # create dataframe from transformed data
        data_trans = data.__class__(index=data.index, data=data_trans)

        # set data
        if self.data.shape[1] == data_trans.shape[1]:
            ds.data.loc[data.index] = data_trans
            self.normalization_method = norm_name
        else:
            raise ValueError("Argument \"normalizer\" should not change the number of features.")

    def reformat_metadata(self, convert_dtypes: bool = False):
        """
        This method performs a basic reformatting of metadata including: Replacing double-spaces with a single space,
        Stripping white space from string ends, Removing mixed-case and capitalizing strings. Additionally one can use
        pandas infer_dtypes function to automatically infer the datatypes for each attribute.

        Args:
            convert_dtypes (bool): Flag for whether or not to infer the datatypes for the metadata. Default is false.


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

        # convert dtypes
        if convert_dtypes:
            self.metadata = self.metadata.convert_dtypes()

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

        # sort data and metadata together to be safe
        self.data.sort_index(inplace=True)
        self.metadata.sort_index(inplace=True)

        # slice data at features
        columns = self.data.columns
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

        # generate slice DataSet
        ds = DataSet(data=deepcopy(data),
                metadata=deepcopy(metadata),
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
        print(textwrap.fill(self._description, line_width))

    # TODO: Add conversion to cdd method.
    # TODO: Add basic normalization methods
    # TODO: Add imputation method
    # TODO: Add varattr

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
            >>> ds = load_dataset(file_path='./iris.ds')
    """
    # open file and unpickle
    with open(file_path, 'rb') as f:
        return pickle.load(f)

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
