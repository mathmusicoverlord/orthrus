# Imports
from dataclasses import dataclass, field
import os
import numpy as np
import pandas as pd

@dataclass
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

    name: str = field(default='')
    save_loc: str = field(default=os.curdir)
    data: np.ndarray = field(default=np.array([]))
    metadata: pd.DataFrame = field(default=pd.DataFrame())

