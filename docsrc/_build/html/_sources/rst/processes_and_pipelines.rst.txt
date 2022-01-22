Processes and Pipelines
=======================

For more involved machine learning experiments we require methods outside
of those provided by the :py:class:`DataSet <orthrus.core.dataset.DataSet>` class,
e.g., :py:meth:`classify() <orthrus.core.dataset.DataSet.classify>`
and :py:meth:`feature_select() <orthrus.core.dataset.DataSet.feature_select>`.
The orthrus package provides two base classes which enable more advanced and
automated workflows.

Processes
---------
The :py:class:`Process <orthrus.core.pipeline.Process>` class is an abstract base class which wraps around
`scikit-learn <https://scikit-learn.org>`_-like objects and makes them directly applicable to
:py:class:`DataSet <orthrus.core.dataset.DataSet>` objects—doing away with much of the boilerplate
code for extracting labels, transforming data, etc. A simple example would be using the 
:py:class:`Transform <orthrus.core.pipeline.Transform>` subclass to log transform a dataset::

    >>> # imports
    >>> import os
    >>> from orthrus.core.pipeline import Transform
    >>> from sklearn.preprocessing import FunctionTransformer
    >>> import numpy as np
    >>> from orthrus.core.dataset import load_dataset
    ...
    >>> # load dataset
    >>> ds = load_dataset(os.path.join(os.environ['ORTHRUS_PATH'],
    ...                                'test_data/Iris/Data/iris.ds'))
    ...
    >>> # define log transform process
    >>> log = Transform(process=FunctionTransformer(np.log),
    ...                 process_name='log',
    ...                 retain_f_ids=True, # keep the original feature names (non-latent)
    ...                 verbosity=1)
    ...
    >>> # run process
    >>> ds, results = log.run(ds)
    ...
    >>> # use resulting transform
    >>> transform = results['batch']['transform']
    >>> ds = transform(ds)
    ...
    >>> # print results
    >>> print(ds.data)
    ----------------------------------------------------------
    Fitting log...
    Transforming the data using log...
            sepal_length  sepal_width  petal_length  petal_width
    0        1.629241     1.252763      0.336472    -1.609438
    1        1.589235     1.098612      0.336472    -1.609438
    2        1.547563     1.163151      0.262364    -1.609438
    3        1.526056     1.131402      0.405465    -1.609438
    4        1.609438     1.280934      0.336472    -1.609438
    ..            ...          ...           ...          ...
    145      1.902108     1.098612      1.648659     0.832909
    146      1.840550     0.916291      1.609438     0.641854
    147      1.871802     1.098612      1.648659     0.693147
    148      1.824549     1.223775      1.686399     0.832909
    149      1.774952     1.098612      1.629241     0.587787
    [150 rows x 4 columns]

It may seem like over-kill for such a simple task, but for pipelines involving
manys steps the overhead is worth it. You may have noticed that the result returned by
the process above is a dictionary of dictionaries. The key in the outer dictionary determines
the **batch** of the dataset, while the inner key determines the type of object returned, e.g.,
a function to transform our data with. In our example
we do not have any train/test splits of the dataset and hence we only have one batch. 
In other cases, where we do have multiple train/test splits, each batch will yield a transform fit
to the training samples of that batch.

Partitioning
^^^^^^^^^^^
The :py:class:`Partition <orthrus.core.pipeline.Partition>` class is used to generate the batches described
above. This class wraps `scikit-learn <https://scikit-learn.org>`_-like paritioners and generates
train/test labels according to the parititoning scheme. For example::
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
        ---------------------------------------------
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

We can see that for each batch we can extract a set of train/test labels which are indexed by the samples
in the dataset. We can also collect all of the train/test labels across all batches by using the
:py:meth:`collapse_results() <orthrus.core.pipeline.Process.collapse_results>` method::
    >>> # collapse train/test labels across batches
    >>> kfold.collapse_results()['tvt_labels']
    --------------------------------------------------------------------------------------
    5-fold-CV splits batch_0_split batch_1_split batch_2_split batch_3_split batch_4_split
    0                        Train          Test         Train         Train         Train
    1                         Test         Train         Train         Train         Train
    2                        Train         Train         Train          Test         Train
    3                         Test         Train         Train         Train         Train
    4                        Train         Train          Test         Train         Train
    ..                         ...           ...           ...           ...           ...
    145                      Train          Test         Train         Train         Train
    146                      Train          Test         Train         Train         Train
    147                      Train         Train          Test         Train         Train
    148                      Train         Train         Train          Test         Train
    149                      Train         Train          Test         Train         Train
    [150 rows x 5 columns]

Train/Validation/Test Made Easy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In some experiments it useful to generate validation data within your training data, e.g.,
hyperparameter tuning. The :py:class:`Partition <orthrus.core.pipeline.Partition>` class can acheive
this task by nesting two of its instances. For example::
    
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

In short, the first parititioning process breaks the dataset into train/test splits, then
the second partitioning process further splits each training set into train/validation splits. This
allows for any sophisticated partitioning of the data into train/validation/test splits. Since each
batch in the original partition will be partitioned itself, we require a 2D index to keep track of
the batches, i.e., **batch_i_j** indicates the **jth** split using partitioner two of the **ith**
split using partitioner one.

Automated Fit and Transform
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now we can observe the real power of using processes vs. directly applying 
`scikit-learn <https://scikit-learn.org>`_-like objects. Say for example we have a situtation where we would
like to perform dimension reduction on our dataset as an intermediate step in a downstream model, e.g., neural net, support vector machines.
Also suppose that we would like to cross-validate our model, but want to adhere to the strictess standards in not allowing
any test data to participate in the dimension reduction process—orthrus has you covered. In this example we will generate train/test splits for
a cross-validation experiment and then train a PCA embedding on each training batch to be used downstream::

    >>> # imports
    >>> import os
    >>> from orthrus.core.dataset import load_dataset
    >>> from orthrus.core.pipeline import Partition
    >>> from sklearn.model_selection import KFold
    >>> from orthrus.core.pipeline import Transform
    >>> from sklearn.decomposition import PCA
    ...
    >>> # load the data
    >>> file_path = os.path.join(os.environ["ORTHRUS_PATH"],
    ...                          "test_data/GSE73072/Data/GSE73072.ds")
    >>> ds = load_dataset(file_path)
    ...
    >>> # define kfold partition
    >>> kfold = Partition(process=KFold(n_splits=5,
    ...                                 shuffle=True,
    ...                                 random_state=124,
    ...                                 ),
    ...                   process_name='5-fold-CV',
    ...                   )
    ...
    >>> # define PCA embedding
    >>> pca = Transform(process=PCA(n_components=4,
    ...                             whiten=True),
    ...                 process_name='pca')
    ...
    >>> # run kfold->pca
    >>> ds, results = pca.run(*kfold.run(ds))
    ------------------------------
    Generating 5-fold-CV splits...
    batch_0:
    Fitting pca...

    batch_1:
    Fitting pca...

    batch_2:
    Fitting pca...

    batch_3:
    Fitting pca...

    batch_4:
    Fitting pca...
    -------------------------------
    >>> # transform data with PCA embedding
    >>> # learned from the second batch of training data
    >>> ds = results['batch_1']['transform'](ds)
    >>> print(ds.data)
    --------------------------------------------------
    Transforming the data using pca...
                   pca_0     pca_1     pca_2     pca_3
    GSM1881744 -0.909890 -1.055641  0.131840 -0.349795
    GSM1881745 -0.897809 -0.777336  0.115019 -0.889299
    GSM1881746 -1.148015 -0.976102  0.264118 -0.383004
    GSM1881747 -1.000370 -0.820823  0.026971 -0.696007
    GSM1881748 -0.980170 -0.542912  0.121559 -0.677571
    ...              ...       ...       ...       ...
    GSM1884625  1.179928 -1.233515  0.262225  0.451030
    GSM1884626  2.965154 -0.131978  1.119863  0.282289
    GSM1884627  0.233108  0.600465  1.285934 -0.129189
    GSM1884628  0.422696 -1.272085  0.866244  0.789034
    GSM1884629  0.649333 -0.078723  2.377741 -0.040473
    [2886 rows x 4 columns]
    --------------------------------------------------

We can even transform the dataset across the transforms generated for each batch with one
call of :py:meth:`transform() <orthrus.core.pipeline.Transform.transform>`, e.g.,

    >>> # transfrom dataset using each batch-transform
    >>> ds_pca_dict = pca.transform(ds)
    -------------------------------------------------------------
    {'batch_0': <orthrus.core.dataset.DataSet at 0x7f834ea11ac0>,
     'batch_1': <orthrus.core.dataset.DataSet at 0x7f834e948250>,
     'batch_2': <orthrus.core.dataset.DataSet at 0x7f834e9e5700>,
     'batch_3': <orthrus.core.dataset.DataSet at 0x7f834ea11dc0>,
     'batch_4': <orthrus.core.dataset.DataSet at 0x7f83be31aee0>}
    --------------------------------------------------------------

Parallel Processing
^^^^^^^^^^^^^^^^^^^
Sometimes training models can be significant cost in time, especially 
in a cross-validation experiment where many models needs to be trained, and
they are done in sequence. Orthrus utilizes the python package
`ray <https://www.ray.io/docs>`_, a distributed computing library,
which enables us to train our models in parallel with minimal code change.
For example in PCA training step above we can start a ray server hosted on
our local machine, and then provide the ``parallel=True`` flag to our
:py:class:`Transform <orthrus.core.pipeline.Transform>` class:

    >>> # initialize the ray server
    >>> import ray
    >>> ray.init()  # specify resources here if needed, see docs.
    ----------------------------------------------------------------------
    {'node_ip_address': '127.0.0.1',
    'raylet_ip_address': '127.0.0.1',
    'redis_address': '127.0.0.1:6379',
    'object_store_address': 'tcp://127.0.0.1:63452',
    'raylet_socket_name': 'tcp://127.0.0.1:63738',
    'webui_url': None,
    'metrics_export_port': 61226,
    'node_id': '0912845b0dea2796c32db069295d89941ba31eb5aebe76cd53ac57f6'}
    ----------------------------------------------------------------------
    >>> # define PCA embedding
    >>> pca = Transform(process=PCA(n_components=4,
    ...                             whiten=True),
    ...                 process_name='pca',
    ...                 parallel=True)
    ...
    >>> # run kfold->pca
    >>> ds, results = pca.run(*kfold.run(ds))
    -------------------------
    (pid=22680) Fitting pca...
    (pid=3396) Fitting pca...
    (pid=8756) Fitting pca...
    (pid=23980) Fitting pca...
    (pid=15592) Fitting pca...
    -------------------------
    >>> # shutdown the server
    >>> ray.shutdown()

Checkout the processes :py:class:`Classify <orthrus.core.pipeline.Classify>`,
:py:class:`Score <orthrus.core.pipeline.Score>`,
and :py:class:`Report <orthrus.core.pipeline.Report>` for more examples.

Pipelines
---------
For workflows involving more than 2 processes, chaining processes
as above can be messy when trying to pass the results of each individual
process along to the next. The 
:py:class:`Pipeline <othrus.core.pipeline.Pipeline>` class provides a way
to seamlessly chain processes together along with the other helpful features,
e.g., checkpointing in long pipelines.Here we provide a simple example of building
a pipeline using the :py:class:`Pipeline <othrus.core.pipeline.Pipeline>` class.

Classifying Tumor Classes from RNA-seq (HiSeq) Samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this experiment we will be working with the PANCAN dataset downloaded from
the `UCI Machine Learning Repository <http://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq#>`_.
To follow along first download the dataset and store the ``data.csv`` and ``labels.csv``
files into the ``$ORTHRUS_PATH/test_data/TCGA-PANCAN-HiSeq-801x20531/Data`` directory.
Once the data is stored their run the script `generate_dataset <../../test_data/TCGA-PANCAN-HiSeq-801x20531/generate_dataset.py>`_

