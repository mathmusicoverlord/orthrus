Visualizing Data
================

In this tutorial we will go over some of the finer points of using the
:py:meth:`visualize() <datasci.core.dataset.DataSet.visualize>` method
of the :py:class:`DataSet <datasci.core.dataset.DataSet>` class. For all
of these examples we will use the GSE73072 dataset, as it contains enough
complexity to demostrate the utitlity of the method, see the
`What is the DataSet class? <what_is_the_dataset.html>`_ tutorial for 
details on how to access this dataset. First we load the dataset::

    # imports
    import os
    from datasci.core.dataset import load_dataset

    # load the data
    file_path = os.path.join(os.environ["DATASCI_PATH"],
                             "test_data/GSE73072/Data/GSE73072.ds")
    ds = load_dataset(file_path)

Basic Usage
-----------
One can easily start plotting their without any in depth knowledge of the method. In this example we will
plot the GSE73072 data using :py:class:`Multi-dimensional Scaling <datasci.manifold.mds.MDS>`

Plotting Backend
------------------
The DataSci package uses two backends for plotting, `Pyplot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html>`_
and `Plotly <https://plotly.com/python/>`_. Pyplot is ideal for generating non-interative plots, such as
figures to be included in a document, while Plotly is ideal for generating interactive plots which can be exported as .html
or hosted on server with use of `dash <https://plotly.com/dash/>`_. We provide a few examples below to demonstrate these two
backends::

    # visualize data using pyplot
    ds.visualize(attr=)

