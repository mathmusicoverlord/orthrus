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


def scatter_pandas(df: pd.DataFrame,
                dim: int,
                grp_colors: str,
                grp_mrkrs: str = None,
                mrkr_size: int = 100,
                mrkr_list: list = None,
                title: str = '',
                x_label: str = '',
                y_label: str = '',
                z_label: str = '',
                sub_title: str = '',
                figsize: tuple = (14, 10),
                no_axes: bool = False,
                save_name: str = None):
    """
    This function uses matplotlib's pyplot to plot the numerical columns of a pandas dataframe against its categorical
    metadata.

    Args:
        df (pandas.DataFrame): DataFrame containing the numerical data and metadata for plotting. The first dim columns
            must contain the numerical data, while the last columns must contain the grp_color attribute and grp_mrkrs
            attribute resp. The grp_mrkrs attribute is optional.

        dim (int): The dimension to plot the data in, it can be 2 or 3.

        grp_colors (str): The name of the column to color the data by.

        grp_mrkrs (str): The name of the column to mark the data by. Mark means to assign
            markers to such as .,+,x,etc..

        mrkr_size (int): The size to be used for the markers. Default is 100.

        mrkr_list (int): List of markers to use for marking the data. The default is a list of
            37 distinct markers.

        title (str): The title of the plot. The default is blank.

        x_label (str): The x-axis label to use. The default is blank.

        y_label (str): The y-axis label to use. The default is blank.

        z_label (str): The z-axis label to use. Only applies if dim = 2. The default is blank.

        sub_title (str): A custom subtitle to the plot. The default is blank.

        figsize (tuple): Tuple whose x-coordinate determines the width of the figure and y-coordinate
            determines the height of the figure. The default is (14, 10).

        no_axes (bool): Flag indicating whether or not to show the axes in the plot.

        save_name (str): The path of where to save the figure. If not given the figure will not be saved.

    Returns:
        inplace method.

    Examples:
        >>> import pandas as pd
        >>> from pydataset import data as pydat
        >>> from datasci.core.dataset import DataSet as DS
        >>> df = pydat('iris')
        >>> data = df[['Sepal.Length', 'Sepal.Width']]
        >>> metadata = df[['Species']]
        >>> ds = DS(name='Iris', data=data, metadata=metadata)
        >>> scatter_pandas(df=df, grp_colors='target', title='Iris Dataset', dim=2, x_label='sepal length(cm)', y_label='sepal width(cm)')
    """

    from matplotlib import pyplot as plt
    from matplotlib.pyplot import Axes
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns

    # set pallete
    num_colors = len(df[grp_colors].unique())
    palette = sns.color_palette("Accent", num_colors)

    # set defaults
    if mrkr_list is None:
        mrkr_list = [".", "+", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "x",
                     "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    if title is None:
        title = ''

    if x_label is None:
        x_label = ''

    if y_label is None:
        y_label = ''

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

                ax.scatter(x, y, z, label=label, c=np.array(palette[i]).reshape(1, -1), marker=mrkr_list[j],
                           s=mrkr_size)

        ax.text2D(0, 0, sub_title, fontsize=16, transform=ax.transAxes)
        ax.set_zlabel(z_label, fontsize=16)


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

                    ax.scatter(x, y, label=label, c=np.array(palette[i]).reshape(1, -1), marker=mrkr_list[j],
                               s=mrkr_size)

        ax.text(0, -.1, sub_title, fontsize=16, transform=ax.transAxes)

    else:
        raise ValueError("Embedding dimension must be 2 or 3!")

    if no_axes:
        ax.axis('off')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=15, title=grp_colors)
    ax.set_title(title, fontsize=15)
    ax.set_ylabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    if not (save_name is None):
        plt.savefig(fname=save_name + '.png', format='png')
    plt.show()

