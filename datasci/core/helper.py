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
        mrkr_list = [".", "+", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "x",
                     "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    if grp_mrkrs is None:
        grp_mrkrs = ''
        df[grp_mrkrs] = ''

    # get dtypes of attrs
    color_type = pd.api.types.infer_dtype(df[grp_colors])
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
                               s=mrkr_size, edgecolors=kwargs.get('edgecolors', 'k'), linewidths=kwargs.get('linewidths', 1))

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
                                   s=mrkr_size, edgecolors=kwargs.get('edgecolors', 'k'), linewidths=kwargs.get('linewidths', 1))

            ax.text(0, -.1, subtitle, fontsize=16, transform=ax.transAxes)

        else:
            raise ValueError("Embedding dimension must be 2 or 3!")

        if grp_mrkrs == '':
            ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=15, title=grp_colors)
        else:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=15, title=grp_colors + '/' + grp_mrkrs)

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
                im = ax.scatter(x, y, z, label=label, c=c, cmap=palette, marker=mrkr_list[j], s=mrkr_size, edgecolors=kwargs.get('edgecolors', 'k'), linewidths=kwargs.get('linewidths', 1))

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
                    im = ax.scatter(x, y, label=label, c=c, cmap=palette, marker=mrkr_list[j], s=mrkr_size, edgecolors=kwargs.get('edgecolors', 'k'), linewidths=kwargs.get('linewidths', 1))

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
            ax.legend(loc='upper left', bbox_to_anchor=(1.1, 0.5), fontsize=15, title=grp_mrkrs)
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
            im = ax.scatter(x, y, z, label=label, c=c, cmap=palette, marker=mrkr_list[0], s=ss * mrkr_size, edgecolors=kwargs.get('edgecolors', 'k'), linewidths=kwargs.get('linewidths', 1))

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
            im = ax.scatter(x, y, label=label, c=c, cmap=palette, marker=mrkr_list[0], s=ss * mrkr_size, edgecolors=kwargs.get('edgecolors', 'k'), linewidths=kwargs.get('linewidths', 1))

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
                           s=ss * mrkr_size, edgecolors=kwargs.get('edgecolors', 'k'), linewidths=kwargs.get('linewidths', 1))

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
                               s=ss * mrkr_size, edgecolors=kwargs.get('edgecolors', 'k'), linewidths=kwargs.get('linewidths', 1))

            ax.text(0, -.1, subtitle, fontsize=16, transform=ax.transAxes)

        else:
            raise ValueError("Embedding dimension must be 2 or 3!")

        ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=15, title=grp_colors)

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
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)
    try:
        ax.set_zlabel(ax.get_zlabel(), fontsize=18)
    except AttributeError:
        pass
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


