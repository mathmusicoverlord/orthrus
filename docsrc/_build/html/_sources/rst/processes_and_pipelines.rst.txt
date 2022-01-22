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
Once the data is stored there run the script `generate_dataset <https://github.com/ekehoe32/orthrus/blob/main/test_data/TCGA-PANCAN-HiSeq-801x20531/generate_dataset.py>`_
to produce the :py:class:`DataSet <orthrus.core.dataset.DataSet>` object. This script
performs some standard preprocessing steps such as filtering out low expression genes,
replacing zero with half the non-zero minimum, and then log2 normalizing the data.

We can visualize the dataset using PCA by running the script
`visualize_dataset <https://github.com/ekehoe32/orthrus/blob/main/test_data/TCGA-PANCAN-HiSeq-801x20531/generate_dataset.py>`_
to produce the plot shown here `TCGA-PANCAN-HiSeq-801x20531_pca_viz_example_4_3d.html <TCGA-PANCAN-HiSeq-801x20531_pca_viz_example_4_3d.html>`_.
We will run a 5-fold cross-validation experiment classifying COAD vs. LUAD tumor classes using:

* Data standardization
* Feature selection (dimension reduction) with :py:class:`SSVMSelect <orthrus.sparse.classifiers.SSVMSelect>`
* Classification with `LinearSVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html>`_

First we load our dataset object:
    >>> # imports
    >>> import os
    >>> from orthrus.core.dataset import load_dataset
    >>> ds = load_dataset(os.path.join(os.environ['ORTHRUS_PATH'],
    ...                                "test_data\\TCGA-PANCAN-HiSeq-801x20531\\Data\\TCGA-PANCAN-HiSeq-801x20531-log2.ds"))

We then restrict our samples to the COAD and LUAD tumor classes:
    >>> # restrict samples
    >>> sample_ids = ds.metadata.query("tumor_class in ['COAD', 'LUAD']").index
    >>> ds = ds.slice_dataset(sample_ids=sample_ids)

and build the processes involved:
    >>> # define kfold partition
    >>> from sklearn.model_selection import KFold
    >>> from orthrus.core.pipeline import Partition
    >>> kfold = Partition(process=KFold(n_splits=5,
    ...                                 shuffle=True,
    ...                                 random_state=3458,
    ...                                 ),
    ...                   process_name='5-fold-CV',
    ...                   )
    ...
    >>> # define standardization
    >>> from sklearn.preprocessing import StandardScaler
    >>> from orthrus.core.pipeline import Transform
    >>> std = Transform(process=StandardScaler(),
    ...                 process_name='std',
    ...                 retain_f_ids=True,
    ...                 )
    ...
    >>> # define feature selector
    >>> from orthrus.solvers.linear import LPPrimalDualPy
    >>> from orthrus.sparse.classifiers.svm import SSVMSelect
    >>> from orthrus.core.pipeline import FeatureSelect
    >>> ssvm = FeatureSelect(process=SSVMSelect(solver=LPPrimalDualPy),
    ...                      process_name='ssvm',
    ...                      supervised_attr='tumor_class',
    ...                      f_ranks_handle='f_ranks'
    ...                      )
    ...
    >>> # define classifier
    >>> from sklearn.svm import LinearSVC
    >>> from orthrus.core.pipeline import Classify
    >>> svc = Classify(process=LinearSVC(class_weight='balanced'),
    ...                process_name='svc',
    ...                class_attr='tumor_class',
    ...                f_weights_handle='coef_'
    ...                )

including the final reporting:
    >>> # define report
    >>> from orthrus.core.pipeline import Report
    >>> report = Report(pred_attr='tumor_class')

We are now ready to build the pipeline:
    >>> # define pipeline
    >>> from orthrus.core.pipeline import Pipeline
    >>> pipeline = Pipeline(processes=(kfold, std, ssvm, svc, report),
    ...                     pipeline_name='kfold_std_ssvm_svc',
    ...                     checkpoint_path=os.path.join(os.environ['ORTHRUS_PATH'],
    ...                                                  "test_data/TCGA-PANCAN-HiSeq-801x20531/" \
    ...                                                  "Pipelines/kfold_std_ssvm_svc.pickle"),
    ...                     parallel=True)

As we can see a pipeline is glorified tuple of processes, 
its primary job is to manage the assembly line and translate
the results from one process to the next. One argument of note is the
:py:attr:`checkpoint_path <orthrus.core.pipeline.Pipeline.checkpoint_path>`, 
which sets the path of the serialized (pickled) pipeline object on the disk,
and enables the pipeline to be saved after the completion of each process. This
is useful in the context of long training and possible interruptions in the computation.
By saving the pipeline periodically, it can be interrupted and pickup after its
last completed process. We can even stop the pipeline before its hits a certain process, and
then pick it up later. We will demonstrate this now:

    >>> # start the ray server
    >>> import ray
    >>> ray.init()
    ...
    >>> # run the pipeline up until the reporting
    >>> # and save the pipeline after each completed process
    >>> pipeline.run(ds, stop_before='report', checkpoint=True)

Notice that our pipeline has been saved to the disk in the location
specified and that we can now start the pipeline again to complete
the final reporting step:

    >>> # complete the pipeline
    >>> pipeline.run(ds, checkpoint=True)

In the excution of this pipeline, we are ensured that all models are only
ever trained on the training data, and we can be sure that our test scores
are as unbiased as possible. We can now extract the test results of our experiment,
this is a good point to show how one loads a pipeline from the disk:

    >>> # load the pipeline (not needed, just for example)
    >>> from orthrus.core.helper import load_object
    >>> pipeline = load_object(os.path.join(os.environ['ORTHRUS_PATH'],
                                            "test_data/TCGA-PANCAN-HiSeq-801x20531/" \
                                            "Pipelines/kfold_std_ssvm_svc.pickle"))

We can now extract our report process from the pipeline to view the test statistics:

    >>> # report test statistics
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> test_scores = report.report()['train_test'].filter(regex="^((?!Support).)*$").filter(regex="Test")
    >>> test_scores.columns = test_scores.columns.str.strip("Test_")
    >>> print("Test Scores:"); print(test_scores)
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Test Scores:
                            Coad_Precision Coad_Recall Coad_F1-scor Luad_Precision  ... Macro avg_F1-scor Weighted avg_Precision Weighted avg_Recall Weighted avg_F1-scor
    report prediction scores                                                         ...
    batch_0_report_scores               1.0         1.0          1.0            1.0  ...               1.0                    1.0                 1.0                  1.0
    batch_1_report_scores               1.0         1.0          1.0            1.0  ...               1.0                    1.0                 1.0                  1.0
    batch_2_report_scores               1.0         1.0          1.0            1.0  ...               1.0                    1.0                 1.0                  1.0
    batch_3_report_scores               1.0         1.0          1.0            1.0  ...               1.0                    1.0                 1.0                  1.0
    batch_4_report_scores               1.0         1.0          1.0            1.0  ...               1.0                    1.0                 1.0                  1.0
    [5 rows x 13 columns]
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

If only every problem was that easy, but it demonstrates the process.
We can also view the features that we found in our feature selection process:

    >>> # show top 10 features for batch 0 against the other batches
    >>> feature_ranks = pipeline.processes[2].collapse_results(which='f_ranks')['f_ranks']
    >>> batch_0_ranks = feature_ranks["batch_0_Ranks_f_ranks"].argsort().values
    >>> feature_ranks.filter(regex='Ranks').iloc[batch_0_ranks[:10]]
    -------------------------------------------------------------------------------------------------------------------------------
    ssvm f_ranks  batch_0_Ranks_f_ranks  batch_1_Ranks_f_ranks  batch_2_Ranks_f_ranks  batch_3_Ranks_f_ranks  batch_4_Ranks_f_ranks
    ssvm f_ranks
    gene_3523                         0                      1                      4                     21                     11
    gene_15899                        1                      0                      0                      6                      0
    gene_4805                         2                      3                      6                     12                     35
    gene_5829                         3                     25                      8                     39                      1
    gene_15591                        4                      9                      3                      3                      8
    gene_3                            5                     18                     17                    118                     19
    gene_14034                        6                     13                      5                     15                     12
    gene_6156                         7                     47                      9                      9                     82
    gene_11349                        8                     17                      1                     61                      4
    gene_10192                        9                     91                     39                     30                     50
    -------------------------------------------------------------------------------------------------------------------------------
    >>> # show attributes for batch 0 features
    >>> batch_0_attrs = feature_ranks.filter(regex='batch_0').iloc[batch_0_ranks]
    ----------------------------------------------------------------------------------------- 
    ssvm f_ranks  batch_0_Ranks_f_ranks  batch_0_absWeights_f_ranks  batch_0_Selected_f_ranks
    ssvm f_ranks
    gene_3523                         0                1.516645e-01                         1
    gene_15899                        1                1.362592e-01                         1
    gene_4805                         2                1.117297e-01                         1
    gene_5829                         3                6.269817e-02                         1
    gene_15591                        4                5.880049e-02                         1
    ...                             ...                         ...                       ...
    gene_10510                    17717                1.801370e-13                         0
    gene_5114                     17718                1.403558e-13                         0
    gene_19791                    17719                1.339517e-13                         0
    gene_13820                    17720                1.130697e-13                         0
    gene_18571                    17721                8.205357e-14                         0

    [17722 rows x 3 columns]
    -----------------------------------------------------------------------------------------

Every process in the pipeline can be accessed for the specific results. See the module
:py:mod:`pipeline <orthrus.core.pipeline>` for more specific details on each process and
pipeline methods.