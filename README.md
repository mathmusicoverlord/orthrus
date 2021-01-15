# DataSci
A collection of python classes and functions for automating pre-processing, visualization, and classification of generic data sets. Read the [docs](https://ekehoe32.github.io/DataSci/)!

## Installing the conda environment
In order to ensure proper behavior of python classes and functions between platforms we recommend installing an isolated ```conda``` environment with the depedencies listed in [environment.yml](environment.yml). To create a new enviroment with these dependencies, from the shell run:
```
conda env create -f environment.yml
```
This will generate the conda environment DataSci and install any dependencies required by the DataSci module. If the user does not have a CUDA >=11 compatible graphics card, then the user can replace [environment.yml](environment.yml) with [environment_nocuda.yml](environment_nocuda.yml). The user can also use their own environment and install the packages listed in either [environment.yml](environment.yml) or [environment_nocuda.yml](environment_nocuda.yml).

## Installing the DataSci package
To install the DataSci package first activate the DataSci environment and then navigate to your local DataSci directory:
```
conda activate DataSci
cd /path/to/DataSci/
```
Fianlly install the package with ```pip```
```
pip install -e .

```
## Basic Usage
The fundamental object in the DataSci package is the DataSet class. Here is an example of loading the iris dataset into the [DataSet](https://ekehoe32.github.io/DataSci/rst/dataset.html#dataset.DataSet) class to create an instance from within the [DataSci](DataSci) directory:

```python
# imports
from datasci.core.dataset import DataSet as DS
import pandas as pd

# load data and metadata
data = pd.read_csv("./test_data/iris_data.csv", index_col=0)
metadata = pd.read_csv("./test_data/iris_metadata.csv", index_col=0)

# create DataSet instance
ds = DS(name='iris', path='./test_data', data=data, metadata=metadata)

```
here ```path``` indicates where ```ds``` will save figures and results outputed by the class methods.
