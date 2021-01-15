# Imports
import os
import numpy as np
import pandas as pd
from numpy.core import ndarray
from pandas.core.frame import DataFrame
from pandas.core.frame import Series

class DataSet:
    """
    Primary base class for storing data and metadata for a generic dataset. Contains methods for quick data
    pre-processing, visualization, and classification.

    Attributes:
        name (str): Reference name for the dataset. Default is the empty string.

        path (str): File path for saving DataSet instance and related outputs. Default is the empty string.

        data (pandas.DataFrame): Numerical data or features of the data set arranged as samples x features.
            Default is the empty DataFrame.

        metadata (pandas.DataFrame): Categorical data or attributes of the dataset arranged as samples x attributes.
            The sample labels in the index column should be the same format as those used for the data DataFrame.
            If labels are missing or there are more labels than in the data, the class will automatically restrict
            to just those samples used in the data and fill in NaN where there are missing samples. Default is the
            empty DataFrame.

        normalization_method (str): Label indicating the normalization used on the data. Future normalization will
            append as normalization_1/normalization_2/.../normalization_n indicating the sequence of normalizations
            used on the data. Default is the empty string.

        imputation_method (str): Label indicating the imputation used on the data. Default is the empty string.
    """

    def __init__(self, name: str = '',
                 path: str = os.curdir,
                 data: DataFrame = pd.DataFrame(),
                 metadata: DataFrame = pd.DataFrame(),
                 normalization_method: str = '',
                 imputation_method: str = ''):

        # Load attributes
        self.name = name
        self.path = path
        self.data = data
        self.metadata = metadata
        self.normalization_method = normalization_method
        self.imputation_method = imputation_method

        # restrict metadata to data
        meta_index = self.metadata.index.drop(self.metadata.index.difference(self.data.index))
        missing_data_index = self.data.index.drop(meta_index)
        missing_data = pd.DataFrame(index=missing_data_index)
        self.metadata = self.metadata.loc[meta_index].append(missing_data)

    def visualize(self, transform,
                  attrname: str,
                  feature_ids: Series = None,
                  sample_ids: Series = None,
                  save: bool = False):
        """

        Args:
            transform (object): Class instance which must contain the method fit_transform.

            attrname (str): Name of the metadata attribute to color samples by.

            feature_ids (pandas.Series): Series indicating

            sample_ids (pandas.Series):

            save:

        Returns:

        """
        pass


    def reformat_metadata(self, convert_dtypes: bool =False):
        """
        Args:
            convert_dtypes (bool): Flag for whether or not to infer the datatypes for the metadata. Default is false.

        This method performs a basic reformatting of metadata including:
            * Replacing double-spaces with a single space
            * Stripping white space from string ends
            * Removing mixed-case and capitalizing strings

        Returns: inplace method.
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


    def autosummarize(self):
        import dash
        import dash_html_components as html
        import dash_core_components as dcc
        from dash.dependencies import Input, Output
        import plotly.graph_objects as go
        import plotly.express as px

        # set the data
        df = self.metadata

        # I've always used the same stylesheet but there are options
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

        # the html.Div should open for html code
        # dcc is dash core component sets up the dash element. The first one here is a dropdown menu
        app.layout = html.Div([html.H1(children=self.name + " Metadata Summarization"),
                               html.Div(children='''Choose an attribute below.''', style={'padding': 10}),

                               html.Div([
                                   dcc.Dropdown(
                                       id='attribute',
                                       options=[{'label': attr, 'value': attr} for attr in df.columns],
                                       value=df.columns[0],
                                       clearable=False)]),

                               html.Div(children='''Choose the number of bins (0 = Auto).''', style={'padding': 10}),

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
                y, x = np.histogram(series[-series.isna()], bins=bins)
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

            fig.update_layout(title_text="Histogram of " + attr + " Attribute", xaxis_title=attr,
                                  yaxis_title="Frequency", title_x=0.5)
            return fig

        app.run_server(debug=True)

        return app

if __name__ == "__main__":
    from datasci.core.DataSet import DataSet as DS
    import pandas as pd

    data = pd.read_csv("/hdd/Test_Data/test_data.csv", index_col=0)
    metadata_more = pd.read_csv("/hdd/Test_Data/test_metadata_more.csv", index_col=0)
    metadata_less = pd.read_csv("/hdd/Test_Data/test_metadata_less.csv", index_col=0)

    ds = DS(name='Test', path='/hdd/Test_Data/', data=data, metadata=metadata_more)
    ds.reformat_metadata(convert_dtypes=True)
    ds.autosummarize()

    print("Hi!")


