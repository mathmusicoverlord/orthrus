'''
This module contains user-defined and general purpose helper functions use by the DataSci package.
'''

from inspect import ismethod
import numpy as np
import pandas as pd
from flask import request


def method_exists(instance: object, method: str):
    """
    This function takes a class instance and a method name and checks whether or not the method name
    is a method of the class instance.

    Args:
        instance (object): Instance of class to check for method.
        method (str): Name of method to check for.

    Returns:
        bool : True if method is a class method for the instance, false otherwise.

    """
    return hasattr(instance, method) and ismethod(getattr(instance, method))


def scatter_pyplot(df: pd.DataFrame,
                   dim: int,
                   grp_colors: str,
                   palette: str = None,
                   grp_mrkrs: str = None,
                   mrkr_list: list = None,
                   subtitle: str = '',
                   figsize: tuple = (14, 10),
                   no_axes: bool = False,
                   save_name: str = None,
                   **kwargs):
    """
    This function uses matplotlib's pyplot to plot the numerical columns of a pandas dataframe against its categorical
    metadata.

    Args:
        df (pandas.DataFrame): DataFrame containing the numerical data and metadata for plotting. The first dim columns
            must contain the numerical data, while the last columns must contain the grp_color attribute and grp_mrkrs
            attribute resp. The grp_mrkrs attribute is optional.

        dim (int): The dimension to plot the data in, it can be 2 or 3.

        grp_colors (str): The name of the column to color the data by.

        palette (str): String signfying the seaborn palette to use. Default is 'Accent'.

        grp_mrkrs (str): The name of the column to mark the data by. Mark means to assign
            markers to such as .,+,x,etc..

        mrkr_list (int): List of markers to use for marking the data. The default is a list of
            37 distinct markers.

        subtitle (str): A custom subtitle to the plot. The default is blank.

        figsize (tuple): Tuple whose x-coordinate determines the width of the figure and y-coordinate
            determines the height of the figure. The default is (14, 10).

        no_axes (bool): Flag indicating whether or not to show the axes in the plot.

        save_name (str): The path of where to save the figure. If not given the figure will not be saved.

        kwargs (dict): All keyword arguments are passed to ``matplotlib.axes.Axes.update()`` if :py:attr:`dim` = 2
            or ``mpl_toolkits.mplot3d.axes3d.Axes3D.update()`` if :py:attr:`dim` = 3.

    Returns:
        inplace method.

    Examples:
        >>> import pandas as pd
        >>> from pydataset import data as pydat
        >>> from datasci.core.helper import scatter_pyplot
        >>> df = pydat('iris')
        >>> scatter_pyplot(df=df,
        ...                grp_colors='Species',
        ...                title='Iris Dataset',
        ...                dim=2,
        ...                xlabel='Sepal Length (cm)',
        ...                ylabel='Sepal Width (cm)')
    """

    from matplotlib import pyplot as plt
    from matplotlib.pyplot import Axes
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns

    # set pallete
    num_colors = len(df[grp_colors].unique())
    if palette is None:
        palette = 'Accent'
    palette = sns.color_palette(palette, num_colors)

    # set defaults
    if mrkr_list is None:
        mrkr_list = [".", "+", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "x",
                     "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    if grp_mrkrs is None:
        grp_mrkrs = str(np.random.rand())
        df[grp_mrkrs] = ''

    if dim == 3:
        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig)
        for i, grp0 in enumerate(df.groupby(grp_colors).groups.items()):
            grp_name0 = grp0[0]
            grp_idx0 = grp0[1]

            for j, grp1 in enumerate(df.groupby(grp_mrkrs).groups.items()):
                grp_name1 = grp1[0]
                grp_idx1 = grp1[1]

                grp_idx = list(set(grp_idx0).intersection(set(grp_idx1)))
                x = df.loc[grp_idx, df.columns[0]]
                y = df.loc[grp_idx, df.columns[1]]
                z = df.loc[grp_idx, df.columns[2]]
                label = grp_name0 + '/' + grp_name1
                label = label.rstrip('/')

                mrkr_size = kwargs.get('s', 100)
                ax.scatter(x, y, z, label=label, c=np.array(palette[i]).reshape(1, -1), marker=mrkr_list[j], s=mrkr_size)

        ax.text2D(0, 0, subtitle, fontsize=16, transform=ax.transAxes)
        #ax.set_zlabel(zlabel, fontsize=16)


    elif dim == 2:
        fig, ax = plt.subplots(1, figsize=figsize)
        for i, grp0 in enumerate(df.groupby(grp_colors).groups.items()):
            grp_name0 = grp0[0]
            grp_idx0 = grp0[1]

            for j, grp1 in enumerate(df.groupby(grp_mrkrs).groups.items()):
                grp_name1 = grp1[0]
                grp_idx1 = grp1[1]

                grp_idx = list(set(grp_idx0).intersection(set(grp_idx1)))
                if len(grp_idx) > 0:
                    x = df.loc[grp_idx, df.columns[0]]
                    y = df.loc[grp_idx, df.columns[1]]
                    label = grp_name0 + '/' + grp_name1
                    label = label.rstrip('/')

                    mrkr_size = kwargs.get('s', 100)
                    ax.scatter(x, y, label=label, c=np.array(palette[i]).reshape(1, -1), marker=mrkr_list[j], s=mrkr_size)

        ax.text(0, -.1, subtitle, fontsize=16, transform=ax.transAxes)

    else:
        raise ValueError("Embedding dimension must be 2 or 3!")

    if no_axes:
        ax.axis('off')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=15, title=grp_colors)
    try:
        kwargs.pop('s')
    except KeyError:
        pass
    ax.update(kwargs)
    #ax.set_title(title, fontsize=15)
    #ax.set_xlabel(xlabel, fontsize=16)
    #ax.set_ylabel(ylabel, fontsize=16)
    if not (save_name is None):
        plt.savefig(fname=save_name + '.png', format='png')
    plt.show()


def scatter_plotly(df: pd.DataFrame,
                   dim: int,
                   grp_colors: str,
                   grp_mrkrs: str = None,
                   mrkr_size: int = 10,
                   mrkr_list: list = None,
                   xlabel: str = '',
                   ylabel: str = '',
                   zlabel: str = '',
                   subtitle: str = '',
                   figsize: tuple = (900, 800),
                   save_name: str = None,
                   use_dash: bool = False,
                   **kwargs):

    """
    This function uses plotly to plot the numerical columns of a pandas dataframe against its categorical
    metadata.

    Args:
        df (pandas.DataFrame): DataFrame containing the numerical data and metadata for plotting. The first dim columns
            must contain the numerical data, while the last columns must contain the grp_color attribute and grp_mrkrs
            attribute resp. The grp_mrkrs attribute is optional.

        dim (int): The dimension to plot the data in, it can be 2 or 3.

        grp_colors (str): The name of the column to color the data by.

        grp_mrkrs (str): The name of the column to mark the data by. Mark means to assign
            markers to such as .,+,x,etc..

        mrkr_size (int): The size to be used for the markers. Default is 10.

        mrkr_list (int): List of markers to use for marking the data. The default is a list of
            37 distinct markers.

        xlabel (str): The x-axis label to use. The default is blank.

        ylabel (str): The y-axis label to use. The default is blank.

        zlabel (str): The z-axis label to use. Only applies if dim = 2. The default is blank.

        subtitle (str): A custom subtitle to the plot. The default is blank.

        figsize (tuple): Tuple whose x-coordinate determines the width of the figure and y-coordinate
            determines the height of the figure. The default is (900, 800).

        save_name (str): The path of where to save the figure. If not given the figure will not be saved.

        use_dash (bool) = Flag indicating whether to host the figure through dash.

        **kwargs (dict): Passed directly to ``plotly.express.scatter`` and then to ``dash.Dash.app.run_server`` for
            configuring host server. See dash documentation for further details.

    Returns:
        inplace method.

    Examples:
        >>> import pandas as pd
        >>> from pydataset import data as pydat
        >>> from datasci.core.helper import scatter_plotly
        >>> df = pydat('iris')
        >>> scatter_plotly(df=df,
        ...                grp_colors='Species',
        ...                dim=2,
        ...                xlabel='Sepal Length (cm)',
        ...                ylabel='Sepal Width (cm)',
        ...                use_dash=True,
        ...                title='Iris Dataset')
    """

    # imports
    import plotly
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    # set defaults
    if xlabel is None:
        xlabel = ''

    if ylabel is None:
        ylabel = ''

    # grab column names
    col0 = df.columns[0]
    col1 = df.columns[1]

    # scatter
    if 'labels' in kwargs.keys():
        raise KeyError("Can not set labels argument for plotly.scatter, must use xlabel, ylabel, etc..")

    if 'symbol' in kwargs.keys():
        raise KeyError("Can not set symbol argument for plotly.scatter, this is defined by cross_attr argument")

    if 'color' in kwargs.keys():
        raise KeyError("Can not set color argument for plotly.scatter, this is defined by attr argument")

    if dim == 2:
        fig = px.scatter(df,
                         x=col0,
                         y=col1,
                         symbol=grp_mrkrs,
                         color=grp_colors,
                         labels={str(col0): xlabel,
                                 str(col1): ylabel},
                         **kwargs)

    elif dim == 3:
        col2 = df.columns[2]
        fig = px.scatter_3d(df,
                            x=col0,
                            y=col1,
                            z=col2,
                            symbol=grp_mrkrs,
                            color=grp_colors,
                            labels={str(col0): xlabel,
                                    str(col1): ylabel,
                                    str(col2): zlabel},
                            **kwargs)
    else:
        raise ValueError("Embedding dimension must be 2 or 3!")

    # update marker size
    fig.update_traces(marker=dict(size=mrkr_size))

    # update markers
    if not (mrkr_list is None):
        for i, label in enumerate(df[grp_mrkrs].unique()):
            fig.for_each_trace(
                lambda trace: trace.update(marker_symbol=mrkr_list[i]) if str(label) in trace.name else ())

    # set title height
    fig.update_layout(title={'y': .92, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})

    # set subtitle
    fig.add_annotation(xref='paper',
                       yref='paper',
                       x=0, y=-.1,
                       showarrow=False,
                       font_size=14,
                       text=subtitle)

    # resize figure
    fig.update_layout(width=figsize[0], height=figsize[1])

    if use_dash:
        import dash
        import dash_html_components as html
        import dash_core_components as dcc
        from dash.dependencies import Input, Output

        # I've always used the same stylesheet but there are options
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

        # the html.Div should open for html code
        # dcc is dash core component sets up the dash element. The first one here is a dropdown menu
        app.layout = html.Div([html.Div([dcc.Graph(figure=fig,
                                                   style={'padding': 20},
                                                   className='row',
                                                   )]),
                               html.Hr(),
                               ])
        for var in scatter_plotly.__code__.co_varnames:
            if var in kwargs.keys():
                kwargs.pop(var)

        for var in plotly.express.scatter.__code__.co_varnames:
            if var in kwargs.keys():
                kwargs.pop(var)

        app.run_server(**kwargs)
    else:
        if not (save_name is None):
            # fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            plotly.offline.plot(fig, filename=(save_name + '.html'))
        else:
            pio.show(fig)
