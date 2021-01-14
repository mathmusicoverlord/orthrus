# Imports
import os
import numpy as np
import pandas as pd

class DataSet:
    """
    Primary base class for storing data and metadata for a generic dataset. Contains methods for quick data
    pre-processing, visualization, and classification.

    Attributes:
        name (str): Reference name for the dataset.

        path (str): File path for saving DataSet instance and related outputs.

        data (numpy.ndarray): Numerical data or features of the data set arranged as samples x features.

        metadata (pandas.DataFrame): Categorical data or attributes of the dataset arranged as samples x attributes
    """

    def __init__(self, name: str = '',
                 path: str = os.curdir,
                 data: np.ndarray = np.array([]),
                 metadata: pd.DataFrame = pd.DataFrame()):

        # Load attributes
        self.name = name
        self.path = path
        self.data = data
        self.metadata = metadata

class Test:
    """
    Test Class.

    Attributes:
        test (test): Test.
    """


