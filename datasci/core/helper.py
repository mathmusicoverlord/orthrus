'''
This module contains user-defined and general purpose helper functions use by the DataSci package.
'''

from inspect import ismethod
import numpy as np
import pandas as pd
import os
import pickle
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

def get_close_matches_icase(word, possibilities, *args, **kwargs):
    """ Case-insensitive version of difflib.get_close_matches """
    import difflib
    import itertools

    lword = word.lower()
    lpos = {}
    for p in possibilities:
        if p.lower() not in lpos:
            lpos[p.lower()] = [p]
        else:
            lpos[p.lower()].append(p)
    lmatches = difflib.get_close_matches(lword, lpos.keys(), *args, **kwargs)
    ret = [lpos[m] for m in lmatches]
    ret = itertools.chain.from_iterable(ret)
    return list(ret)

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
                   block=True,
                   **kwargs):
    """
    This function uses matplotlib's pyplot to plot the numerical columns of a pandas dataframe against its categorical
    or numerical metadata.

    Args:
        df (pandas.DataFrame): DataFrame containing the numerical data and metadata for plotting. The first dim columns
            must contain the numerical data, while the last columns must contain the grp_color attribute and grp_mrkrs
            attribute resp. The grp_mrkrs attribute is optional.

        dim (int): The dimension to plot the data in, it can be 2 or 3.

        grp_colors (str): The name of the column to color the data by.

        palette (str): String signfying the seaborn palette to use. Default is 'Accent' for categorical metadata, and
            'magma' for numerical metadata.

        grp_mrkrs (str): The name of the column to mark the data by. Mark means to assign
            markers to such as .,+,x,etc..

        mrkr_list (int): List of markers to use for marking the data. The default is a list of
            37 distinct markers.

        subtitle (str): A custom subtitle to the plot. The default is blank.

        figsize (tuple): Tuple whose x-coordinate determines the width of the figure and y-coordinate
            determines the height of the figure. The default is (14, 10).

        no_axes (bool): Flag indicating whether or not to show the axes in the plot.

        save_name (str): The path of where to save the figure. If not given the figure will not be saved.

        block (bool): Passed to pyplot's show function. If True the user must close the pervious plot before
            another plot will appear.

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
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns

    # set defaults
    if mrkr_list is None:
        mrkr_list = ["^", "o", ",", ".", "v", "+", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "x",
                     "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    if grp_mrkrs is None:
        grp_mrkrs = ''
        df[grp_mrkrs] = ''

    # get dtypes of attrs
    color_type = pd.api.types.infer_dtype(df.loc[~df[grp_colors].isna(), grp_colors])
    mrkr_type = pd.api.types.infer_dtype(df[grp_mrkrs])

    if color_type != 'floating' and mrkr_type != 'floating':
        # data is discrete

        # set pallete
        num_colors = len(df[grp_colors].unique())
        if palette is None:
            palette = 'Accent'
        palette = sns.color_palette(palette, num_colors)

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
                    label = str(grp_name0) + '/' + str(grp_name1)
                    label = label.rstrip('/')

                    mrkr_size = kwargs.get('s', 100)
                    ax.scatter(x, y, z, label=label, c=np.array(palette[i]).reshape(1, -1), marker=mrkr_list[j],
                               s=mrkr_size, alpha=kwargs.get('alpha', 1), edgecolors=kwargs.get('edgecolors', 'face'), linewidths=kwargs.get('linewidths', 1))

            ax.text2D(0, 0, subtitle, fontsize=16, transform=ax.transAxes)
            # ax.set_zlabel(zlabel, fontsize=16)

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
                        label = str(grp_name0) + '/' + str(grp_name1)
                        label = label.rstrip('/')

                        mrkr_size = kwargs.get('s', 100)
                        ax.scatter(x, y, label=label, c=np.array(palette[i]).reshape(1, -1), marker=mrkr_list[j],
                                   s=mrkr_size, alpha=kwargs.get('alpha', 1), edgecolors=kwargs.get('edgecolors', 'face'), linewidths=kwargs.get('linewidths', 1))

            ax.text(0, -.1, subtitle, fontsize=16, transform=ax.transAxes)

        else:
            raise ValueError("Embedding dimension must be 2 or 3!")

        if grp_mrkrs == '':
            lg = ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=15, title=grp_colors)
        else:
            lg = ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=15, title=grp_colors + '/' + grp_mrkrs)

    if color_type == 'floating' and mrkr_type != 'floating':
        # color is continuous

        # set pallete
        if palette is None:
            palette = 'magma'
        palette = sns.color_palette(palette, as_cmap=True)
        # df['c'] = (df[grp_colors] - df[grp_colors].min())/df[grp_colors].max()
        df['c'] = df[grp_colors]

        if dim == 3:
            fig = plt.figure(figsize=figsize)
            ax = Axes3D(fig)

            for j, grp1 in enumerate(df.groupby(grp_mrkrs).groups.items()):
                grp_name1 = grp1[0]
                grp_idx1 = grp1[1]

                x = df.loc[grp_idx1, df.columns[0]]
                y = df.loc[grp_idx1, df.columns[1]]
                z = df.loc[grp_idx1, df.columns[2]]
                c = df.loc[grp_idx1, 'c']
                label = grp_name1

                mrkr_size = kwargs.get('s', 100)
                im = ax.scatter(x, y, z, label=label, c=c, cmap=palette, marker=mrkr_list[j], s=mrkr_size, alpha=kwargs.get('alpha', 1), edgecolors=kwargs.get('edgecolors', 'face'), linewidths=kwargs.get('linewidths', 1))

            if grp_mrkrs != '':
                cbar = plt.colorbar(im, ax=ax, location='left', shrink=.6, pad=-0.0001)
                cbar.set_label(grp_colors, rotation=90)
            else:
                cbar = plt.colorbar(im, ax=ax, shrink=.6, pad=-0.1)
                cbar.set_label(grp_colors, rotation=270, labelpad=10)

            ax.text2D(0, 0, subtitle, fontsize=16, transform=ax.transAxes)
            # ax.set_zlabel(zlabel, fontsize=16)

        elif dim == 2:
            fig, ax = plt.subplots(1, figsize=figsize)

            for j, grp1 in enumerate(df.groupby(grp_mrkrs).groups.items()):
                grp_name1 = grp1[0]
                grp_idx1 = grp1[1]

                if len(grp_idx1) > 0:
                    x = df.loc[grp_idx1, df.columns[0]]
                    y = df.loc[grp_idx1, df.columns[1]]
                    c = df.loc[grp_idx1, 'c']
                    label = grp_name1

                    mrkr_size = kwargs.get('s', 100)
                    im = ax.scatter(x, y, label=label, c=c, cmap=palette, marker=mrkr_list[j], s=mrkr_size, alpha=kwargs.get('alpha', 1), edgecolors=kwargs.get('edgecolors', 'face'), linewidths=kwargs.get('linewidths', 1))

            if grp_mrkrs != '':
                cbar = fig.colorbar(im, ax=[ax], location='left', shrink=1, pad=0.1)
                cbar.set_label(grp_colors, rotation=90)
            else:
                cbar = fig.colorbar(im, ax=[ax], shrink=1, pad=-0.15)
                cbar.set_label(grp_colors, rotation=270, labelpad=10)

            ax.text(0, -.1, subtitle, fontsize=16, transform=ax.transAxes)
        else:
            raise ValueError("Embedding dimension must be 2 or 3!")

        if grp_mrkrs != '':
            lg = ax.legend(loc='upper left', bbox_to_anchor=(1.1, 0.5), fontsize=15, title=grp_mrkrs)
            leg = ax.get_legend()
            for handle in leg.legendHandles:
                handle.set_color('black')
                handle.set_alpha(1)

    if color_type == 'floating' and mrkr_type == 'floating':
        # color is continuous

        # set pallete
        if palette is None:
            palette = 'magma'
        palette = sns.color_palette(palette, as_cmap=True)
        df['c'] = (df[grp_colors] - df[grp_colors].min()) / df[grp_colors].max()
        ss = (df[grp_mrkrs] - df[grp_mrkrs].min())
        ss = 24 * (ss / ss.max()) + 1
        ss = ss.values

        if dim == 3:
            fig = plt.figure(figsize=figsize)
            ax = Axes3D(fig)

            x = df[df.columns[0]]
            y = df[df.columns[1]]
            z = df[df.columns[2]]
            c = df['c']
            label = ''

            mrkr_size = kwargs.get('s', 100)
            im = ax.scatter(x, y, z, label=label, c=c, cmap=palette, marker=mrkr_list[0], s=ss * mrkr_size, alpha=kwargs.get('alpha', 1), edgecolors=kwargs.get('edgecolors', 'face'), linewidths=kwargs.get('linewidths', 1))

            cbar = plt.colorbar(im, ax=ax, shrink=.6, pad=-0.1)
            cbar.set_label(grp_colors, rotation=270, labelpad=10)

            ax.text2D(0, 0, subtitle, fontsize=16, transform=ax.transAxes)
            # ax.set_zlabel(zlabel, fontsize=16)

        elif dim == 2:
            fig, ax = plt.subplots(1, figsize=figsize)

            x = df[df.columns[0]]
            y = df[df.columns[1]]
            c = df['c']
            label = ''

            mrkr_size = kwargs.get('s', 100)
            im = ax.scatter(x, y, label=label, c=c, cmap=palette, marker=mrkr_list[0], s=ss * mrkr_size, alpha=kwargs.get('alpha', 1), edgecolors=kwargs.get('edgecolors', 'face'), linewidths=kwargs.get('linewidths', 1))

            cbar = plt.colorbar(im, ax=[ax], shrink=1, pad=-0.15)
            cbar.set_label(grp_colors, rotation=270, labelpad=10)

            ax.text(0, -.1, subtitle, fontsize=16, transform=ax.transAxes)

        else:
            raise ValueError("Embedding dimension must be 2 or 3!")

    if color_type != 'floating' and mrkr_type == 'floating':

        # set pallete
        num_colors = len(df[grp_colors].unique())
        if palette is None:
            palette = 'Accent'
        palette = sns.color_palette(palette, num_colors)

        ss = (df[grp_mrkrs] - df[grp_mrkrs].min())
        ss = 24 * (ss / ss.max()) + 1
        df['s'] = ss

        if dim == 3:
            fig = plt.figure(figsize=figsize)
            ax = Axes3D(fig)
            for i, grp0 in enumerate(df.groupby(grp_colors).groups.items()):
                grp_name0 = grp0[0]
                grp_idx0 = grp0[1]

                x = df.loc[grp_idx0, df.columns[0]]
                y = df.loc[grp_idx0, df.columns[1]]
                z = df.loc[grp_idx0, df.columns[2]]
                ss = df.loc[grp_idx0, 's']
                label = grp_name0

                mrkr_size = kwargs.get('s', 100)
                ax.scatter(x, y, z, label=label, c=np.array(palette[i]).reshape(1, -1), marker=mrkr_list[0],
                           s=ss * mrkr_size, alpha=kwargs.get('alpha', 1), edgecolors=kwargs.get('edgecolors', 'face'), linewidths=kwargs.get('linewidths', 1))

            ax.text2D(0, 0, subtitle, fontsize=16, transform=ax.transAxes)
            # ax.set_zlabel(zlabel, fontsize=16)

        elif dim == 2:
            fig, ax = plt.subplots(1, figsize=figsize)
            for i, grp0 in enumerate(df.groupby(grp_colors).groups.items()):
                grp_name0 = grp0[0]
                grp_idx0 = grp0[1]

                if len(grp_idx0) > 0:
                    x = df.loc[grp_idx0, df.columns[0]]
                    y = df.loc[grp_idx0, df.columns[1]]
                    ss = df.loc[grp_idx0, 's']
                    label = grp_name0

                    mrkr_size = kwargs.get('s', 100)
                    ax.scatter(x, y, label=label, c=np.array(palette[i]).reshape(1, -1), marker=mrkr_list[0],
                               s=ss * mrkr_size, alpha=kwargs.get('alpha', 1), edgecolors=kwargs.get('edgecolors', 'face'), linewidths=kwargs.get('linewidths', 1))

            ax.text(0, -.1, subtitle, fontsize=16, transform=ax.transAxes)

        else:
            raise ValueError("Embedding dimension must be 2 or 3!")

        lg = ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=15, title=grp_colors)


    if no_axes:
        ax.axis('off')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    try:
        kwargs.pop('s')
    except KeyError:
        pass
    try:
        kwargs.pop('edgecolors')
    except KeyError:
        pass
    try:
        kwargs.pop('linewidths')
    except KeyError:
        pass
    ax.update(kwargs)
    # ax.set_title(title, fontsize=15)
    ax.set_title(ax.get_title(), fontsize=16)
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)
    try:
        ax.set_zlabel(ax.get_zlabel(), fontsize=18)
    except AttributeError:
        pass
    try:
        lg.get_title().set_fontsize(18)
    except UnboundLocalError:
        pass
    if not (save_name is None):
        plt.savefig(fname=save_name + '.png', format='png')
    plt.rc('legend', **{'fontsize': 16})
    plt.show(block=block)

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
    or numerical metadata.

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

        use_dash (bool): Flag indicating whether to host the figure through dash.

        **kwargs (dict): Passed directly to ``plotly.express.scatter`` and then to ``dash.Dash.app.run_server`` for
            configuring host server. See dash documentation for further details.

    Returns:
        fig: The figure object for more advanced plots

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

    # sort dataframe by color
    df = df.iloc[df[grp_colors].argsort().values]

    # set defaults
    if xlabel is None:
        xlabel = ''

    if ylabel is None:
        ylabel = ''

    # grab column names
    col0 = df.columns[0]
    col1 = df.columns[1]

    # convert int to str for discrete labels
    if df[grp_colors].dtype == 'int64':
        df = df.astype({grp_colors: str})

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
                         **{k: kwargs[k] for k in plotly.express.scatter.__code__.co_varnames if k in kwargs})

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
                            **{k: kwargs[k] for k in plotly.express.scatter.__code__.co_varnames if k in kwargs})
    else:
        raise ValueError("Embedding dimension must be 2 or 3!")

    # update marker size
    fig.update_traces(marker=dict(size=mrkr_size))

    # update markers
    if not (mrkr_list is None):
        for i, label in enumerate(df[grp_mrkrs].unique()):
            fig.for_each_trace(
                lambda trace: trace.update(marker=dict(symbol=mrkr_list[i])) if str(label) in trace.name else ())

    # set title height
    fig.update_layout(title={'y': .92, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, coloraxis_colorbar=dict(yanchor="top", y=1, x=-.2,
                                          ticks="outside"))

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

def plot_scores(results_list, param_list=None, average='mean', variation='std', figsize=(20, 10), **kwargs):
    """
    This function plots the training and test scores from the results of classification experiments over a
    continuous range of hyper-parameters. It is helpful for these accuracy scores as one varies a continuous parameter
    for the classifier, or experiment.

    Args:
        results_list (list): Each item in the list must be a dictionary which has a "scores" key pointing to a dataframe
            containing "Test" and "Train" rows of accuracy scores. See the output of :py:meth:`DataSet.classify`.

        param_list (list or ndarray): List of hyper-parameters used to generate each result in the ``results_list``.
            If ``None`` each result will be indexed as 1,2,3,...

        average (string): Method of averaging to use. So far there are only two options: "mean" and "median". The
            deafult is "mean".

        variation (string): Method of variation to use; provides error bars in the plot to see the variation of the
            scores across experiments. So far there are only two options: "std" and "minmax", where "minmax" indicates
            using the minimum score and maximum score respectively. The default is "std".

        figsize (tuple): Size of the figure, e.g. (width, height). The default is (20, 10).

        **kwargs (dict): All keyword arguments are passed to ``matplotlib.axes.Axes.update()`` for plot customizations.
            See `here <https://matplotlib.org/3.3.3/api/axes_api.html>`_ for all possible inputs.
    Returns:
        inplace method.

    Examples:
            >>> # imports
            >>> import datasci.core.dataset as dataset
            >>> from datasci.sparse.classifiers.svm import SSVMClassifier as SSVM
            >>> from calcom.solvers import LPPrimalDualPy
            >>> from sklearn.model_selection import KFold
            >>> from sklearn.metrics import balanced_accuracy_score as bsr
            >>> from datasci.core.helper import plot_scores
            ...
            >>> # load dataset
            >>> ds = dataset.load_dataset('./test_data/GSE161731_tmm_log2.ds')
            ...
            >>> # setup classification experiment
            >>> ssvm = SSVM(solver=LPPrimalDualPy, use_cuda=True)
            >>> kfold = KFold(n_splits=5, shuffle=True, random_state=0)
            >>> covid_healthy = ds.metadata[attr].isin(['COVID-19', 'healthy'])
            ...
            >>> # Run classification while varying C in SSVM
            >>> C_range = np.arange(1e-2, .5, 1e-2)
            >>> results_list = []
            >>> for C in C_range:
            >>>     ssvm.C = C
            >>>     # run classification
            >>>     results = ds.classify(classifier=ssvm,
            ...                           attr='cohort',
            ...                           sample_ids=covid_healthy,
            ...                           partitioner=kfold,
            ...                           scorer=bsr,
            ...                           experiment_name='covid_vs_healthy_SSVM_5-fold',
            ...                           )
            >>>     results_list.append(results)
            ...
            >>> # plot scores across C_range
            >>> plot_scores(results_list,
            ...             param_list=C_range,
            ...             title='Mean BSR of 5-fold SSVM /w Std. Dev. Err Bars',
            ...             ylim=[0, 1],
            ...             yticks=np.arange(0, 1.1, .1),
            ...             xlabel='C',
            ...             ylabel='balanced success rate')

    """
    # imports
    import matplotlib.pyplot as plt

    # intitialize outputs
    test_averages = []
    test_variations_l = []
    test_variations_h = []
    train_averages = []
    train_variations_l= []
    train_variations_h = []

    for results in results_list:

        if average == 'mean':
            train_average = results['scores'].mean(axis=1)['Train']
            test_average = results['scores'].mean(axis=1)['Test']
        elif average == 'median':
            train_average = results['scores'].median(axis=1)['Train']
            test_average = results['scores'].median(axis=1)['Test']

        train_averages.append(train_average)
        test_averages.append(test_average)

        if variation == 'std':
            train_variation_l = train_average - results['scores'].std(axis=1)['Train']
            train_variation_h = train_average + results['scores'].std(axis=1)['Train']
            test_variation_l = test_average - results['scores'].std(axis=1)['Test']
            test_variation_h = test_average + results['scores'].std(axis=1)['Test']

        elif variation == 'minmax':
            train_variation_l = results['scores'].min(axis=1)['Train']
            train_variation_h = results['scores'].max(axis=1)['Train']
            test_variation_l = results['scores'].min(axis=1)['Test']
            test_variation_h = results['scores'].max(axis=1)['Test']

        train_variations_l.append(train_variation_l)
        train_variations_h.append(train_variation_h)
        test_variations_l.append(test_variation_l)
        test_variations_h.append(test_variation_h)


    # convert lists to ndarrays
    train_averages = np.array(train_averages)
    test_averages = np.array(test_averages)
    train_variations_l = np.array(train_variations_l)
    train_variations_h = np.array(train_variations_h)
    test_variations_l = np.array(test_variations_l)
    test_variations_h = np.array(test_variations_h)

    # check for missing parameters list
    if param_list is None:
        param_list = np.arange(1, len(train_averages)+1)
    elif type(param_list) == list:
        param_list = np.array(param_list)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.plot(param_list, train_averages, "b", label='Train')
    ax.fill_between(param_list, train_variations_l, train_variations_h, color="b", alpha=0.2)
    ax.plot(param_list, test_averages, "r", label='Test')
    ax.fill_between(param_list, test_variations_l, test_variations_h, color="r", alpha=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=15)
    ax.update(kwargs)
    plt.show()

def generate_project(name: str, file_path: str):
    """
    This function creates the directory structure for a project— this includes a Data, Experiments, and scripts directory.

    Args:
        name (str): The name of the project.
        file_path (str): The file path to the location where the project directory will be created.

    Returns:
        inplace
    """
    import os

    # define the project directory
    proj_dir = os.path.join(file_path, name)
    os.makedirs(proj_dir, exist_ok=True)

    # define data directory
    data_dir = os.path.join(proj_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True)

    # define experiments directory
    exps_dir = os.path.join(proj_dir, 'Experiments')
    os.makedirs(exps_dir, exist_ok=True)

    # define data directory
    scripts_dir = os.path.join(proj_dir, 'Scripts')
    os.makedirs(scripts_dir)

def generate_experiment(name: str, proj_dir: str):
    """
    This function creates the directory structure for an experiment and generates a parameters python file containing
    a template for experimental parameters to be exported for the experiment in mind. The experiment will automatically
    be placed in the Experiments directory of the project directory.

    Args:
        name (str): The name of the experiment.
        proj_dir (str): The file path of the project directory where the data is held. See :py:func:`generate_project`
            for auto-generation of a project directory.

    Returns:
        inplace
    """
    import os

    # create experiment directory
    if 'Experiments' not in os.path.basename(os.path.abspath(proj_dir)):
        exps_dir = os.path.join(proj_dir, 'Experiments')
    else:
        exps_dir = os.path.abspath(proj_dir)
        proj_dir = os.path.abspath(os.path.dirname(exps_dir))
    exp_dir = os.path.join(exps_dir, name)
    os.makedirs(exps_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    # create figures and results directories
    fig_dir = os.path.join(exp_dir, 'Figures')
    results_dir = os.path.join(exp_dir, 'Results')
    os.makedirs(fig_dir)
    os.makedirs(results_dir)

    # generate python file
    params_file = os.path.join(exp_dir, '_'.join([name, 'params.py']))
    params_text = "\"\"\"\nThis file contains the experimental constants for the experiment " + name + ".\n" \
                  "All experimental parameters to be exported are denoted by UPPERCASE names as a convention.\n" \
                  "\"\"\"\n\n" \
                  "# imports\n" \
                  "import datetime\n" \
                  "import os\n" \
                  "from datasci.core.dataset import load_dataset\n\n" \
                  "# set experiment name\n" \
                  "EXP_NAME = \'" + name + "\'\n\n" \
                  "# set working directories\n" \
                  "PROJ_DIR = \'" + proj_dir + "\'  # <---- put your absolute path\n" \
                  "DATA_DIR = os.path.join(PROJ_DIR, \'Data\')\n" \
                  "EXP_DIR = os.path.join(PROJ_DIR, \'Experiments\', EXP_NAME)\n" \
                  "RESULTS_DIR = os.path.join(EXP_DIR, \'Results\')\n\n" \
                  "# generate figures directory by date\n" \
                  "dt = datetime.datetime.now()\n" \
                  "dt = datetime.date(dt.year, dt.month, dt.day)\n" \
                  "FIG_DIR = os.path.join(EXP_DIR, \'Figures\', dt.__str__())\n" \
                  "os.makedirs(FIG_DIR, exist_ok=True)\n\n" \
                  "# load dataset\n" \
                  "DATASET = load_dataset(os.path.join(DATA_DIR, \'dataset.ds\'))\n" \
                  "DATASET.path = FIG_DIR\n\n" \
                  "# restrict samples\n" \
                  "SAMPLE_IDS = DATASET.metadata.query(\'query here\').index\n\n" \
                  "# restrict features\n" \
                  "FEATURE_IDS = DATASET.vardata.query(\'query here\').index\n\n" \
                  "# other parameters"

    with open(params_file, "w") as f:
        f.write(params_text)
    return

def module_from_path(module_name:str, module_path: str):
    """
    This function imports a module from a file path and returns the module object.

    Args:
        module_name (string): Name of the module.
        module_path (string): Path of the module

    Returns:
        object : The module pointed to by the file path.
    """
    # imports
    import importlib
    import sys

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module

def default_val(module, attr: str, val=None):
    """
    Returns a default value when a module doesn't contain an attribute.
    Args:
        module: Module in consideration
        attr (str): The name of the attribute whose existence is in question.
        val: The value to be used in the case this attribute doesn't exist. The default is None.

    Returns:
        The value of the attribute or the default value.
    """

    if hasattr(module, attr):
        return eval("module." + attr)
    else:
        return val

def load_object(file_path: str, block=True):
    """
    This function loads and returns any object stored in pickle format at the file_path.

    Args:
        file_path (str): Path of the file to load the instance from.

        block (bool): If False and the file is not found, the function will return None. The default is True, so
            the function will error when the file is not found.

    Returns:
        Pickle object stored at the file_path.

    Examples:
            >>> ifr = load_object(file_path='./tol_vs_res_liver_ifr.pickle')
    """

    if block:
        # open file and unpickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        try:
            # open file and unpickle
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

def save_object(object, file_path):
    """
    This method saves an an object in pickle format at the specified path.

    Args:
        file_path (str): Path of the file to save the instance to.

    Returns:
        inplace method.

    """
    # pickle class instance
    with open(file_path, 'wb') as f:
        pickle.dump(object, file=f)

def generate_save_path(file_path: str, overwrite: bool = False):
    """
    This function takes a file path, checks if the file exists, and then
    appends an integer in parentheses to the file name depending on the
    number of copies. This mimics the Linux functionality of making copies.
    If overwrite is True, then the function just returns the original path.

    Args:
        file_path (str): The file path to be checked.

        overwrite (bool): Flag indicating whether or not to overwrite the file.

    Returns:
        str : The modified file path.
    """
    # convert save_path to correct format
    file_path = os.path.abspath(file_path)

    if not overwrite:
        # check if file exists
        exists = os.path.isfile(file_path)
        i = 0

        # find copy number
        while exists:
            i += 1
            path_comps = file_path.split('.')
            end_str = path_comps[-2]
            end_str = ''.join([end_str, '(', str(i), ')'])
            path_comps[-2] = end_str
            file_path = os.path.abspath('.'.join(path_comps))
            exists = os.path.isfile(file_path)

    return file_path


def batch_jobs_(function_handle, 
                list_of_arguments, 
                verbose_frequency : int=10,
                num_cpus_per_worker : float=1., 
                num_gpus_per_worker : float=0.,
                local_mode=False):

    """
    This methods creates and manages batch jobs to be run in parallel. The method takes a function_handle,
    which defines the worker, and a list of arguments for the jobs.

    Args:
        function_handle: Handle of the function or job

        list of arguments (list of list): It is a list of argument list (see example below).

        verbose_frequency (int) : this parameter controls the frequency of progress outputs for the ray workers to console; an output is 
            printed to console after every verbose_frequency number of processes complete execution. (default: 10)
        
        num_cpus_per_worker (float) : Number of CPUs each worker needs. This can be a fraction, check 
            `ray specifying required resources <https://docs.ray.io/en/master/walkthrough.html#specifying-required-resources>`_ for more details. (default: 1.)

        num_gpus_per_worker (float) : Number of GPUs each worker needs. This can be fraction, check 
            `ray specifying required resources <https://docs.ray.io/en/master/walkthrough.html#specifying-required-resources>`_ for more details. (default: 0.)

    Return:
        a list of Ray process object references for the all jobs that were executed in parallel (all have finished execution).
        Note: This method calls ray.init() but doesn't call ray.shutdown() to preseve object references. It must be done
           after the object references have been used

    Example:
        >>> import ray
        >>> from datasci.core.helper import batch_jobs_
        >>> import numpy as np
        >>> @ray.remote
        ... def job_handle(a: int, b: int):
        ...     return a + b
        >>> list_of_args = []
        >>> for i in range(100):
        ...     a = np.random.randint(200)
        ...     b = np.random.randint(200)
        ...     args = [a, b]
        ...     list_of_args.append(args)
        >>> process_refs = batch_jobs_(job_handle, list_of_args, verbose_frequency=10, num_cpus_per_worker=0.5)
        >>> for process in process_refs:
        ...     print(ray.get(process))
        >>> ray.shutdown()
    """
    import ray
    ray.init(ignore_reinit_error=True, local_mode=local_mode)

    #calculate the max number of processes to run at one time
    import multiprocessing
    from math import floor
    num_cpus = ray.available_resources()['CPU']
    max_cpu_workers = floor(num_cpus / num_cpus_per_worker)

    try:
        num_gpus = ray.available_resources()['GPU']
    except: 
        num_gpus = 0

    if num_gpus_per_worker!=0:
        max_gpu_workers = floor(num_gpus / num_gpus_per_worker)
    else:
        max_gpu_workers = np.inf
    
    max_processes = max_cpu_workers if max_cpu_workers < max_gpu_workers else max_gpu_workers
    total_processes = len(list_of_arguments)
    num_running=0
    num_finished=0
    processes = []
    all_processes = []
    i = 1

    #change resource requirements of the worker
    #function_handle.options(num_cpus=num_cpus_per_worker, num_gpus=num_gpus_per_worker)
    #options call is not working, setting them manually
    function_handle._num_cpus = num_cpus_per_worker
    function_handle._num_gpus = num_gpus_per_worker
    for arguments in list_of_arguments:
        if num_finished >= i * verbose_frequency:
            print('%d of %d processes finished'%(num_finished, total_processes))
            i = i+1
        if num_running == max_processes:
            finished_processes, processes = ray.wait(processes)
            num_running -=len(finished_processes)
            num_finished +=len(finished_processes)

        future = function_handle.remote(*arguments)
        processes.append(future)
        all_processes.append(future)
        num_running+=1

    #wait for all remaining processes to finish
    finished_processes, processes = ray.wait(processes, num_returns=len(processes))
    num_running -=len(finished_processes)
    num_finished +=len(finished_processes)
    print('%d of %d processes finished'%(num_finished, total_processes))

    assert num_finished == total_processes, 'All processes were not processed. %d processes are unaccounted for.' \
                                        % (total_processes - num_finished)
    assert num_running == 0, '%d processes still running' % num_running
    return all_processes


def pop_first_element(x):
    """
    Pops and returns the first element from an iterator.
    If the object is not an iterator the object itself is returned.

    Args:
        x (object): object to be popped.

    Returns:
        object: An element in x or x itself.
    """

    try:
        x = x.__next__()
    except:
        pass

    return x
