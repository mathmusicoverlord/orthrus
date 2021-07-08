# DataSci
A collection of python classes and functions for automating pre-processing, visualization, classification, and features selection for generic data sets. Read the [docs](https://ekehoe32.github.io/DataSci/)!

## Installing the conda environment
In order to ensure proper behavior of python classes and functions between platforms we recommend installing an isolated ```conda``` environment with the depedencies listed in [environment.yml](environment.yml). To create a new enviroment with these dependencies, from the shell run:
```bash
conda env create -f environment.yml
```
This will generate the conda environment DataSci and install any dependencies required by the DataSci module. If the user does not have a CUDA >=11 compatible graphics card, then the user can replace [environment.yml](environment.yml) with [environment_nocuda.yml](environment_nocuda.yml). The user can also use their own environment and install the packages listed in either [environment.yml](environment.yml) or [environment_nocuda.yml](environment_nocuda.yml).

## Installing the DataSci package
To install the DataSci package first activate the DataSci environment and then navigate to your local DataSci directory:
```bash
conda activate DataSci
cd /path/to/DataSci/
```
Install the package with ```pip```
```bash
pip install -e .
```
Finally add ```DATASCI_PATH=/path/to/DataSci/``` to your environment variables (different for each OS).

## Basic Usage
The fundamental object in the DataSci package is the DataSet class. Here is an example of loading the iris dataset into the [DataSet](https://ekehoe32.github.io/DataSci/rst/datasci.core.html#datasci.core.dataset.DataSet) class to create an instance from within the DataSci directory:

```python
# imports
from datasci.core.dataset import DataSet as DS
import pandas as pd

# load data and metadata
data = pd.read_csv("test_data/Iris/Data/iris_data.csv", index_col=0)
metadata = pd.read_csv("test_data/Iris/Data/iris_metadata.csv", index_col=0)

# create DataSet instance
ds = DS(name='iris', path='./test_data', data=data, metadata=metadata)

# save dataset
ds.save()

```
here ```path``` indicates where ```ds``` will save figures and results output by the class methods.

## Creating a Project Environment
To increase organization and reproducibility of results the DataSci package includes helper functions for generating a project directory and experiment subdirectories. Here is an example where we create a project directory called *Iris* and then generate an experiment directory called *setosa_versicolor_classify_species_svm* where we intend to classify setosa and versicolor species with an SVM classifier.

```python
# imports
from datasci.core.helper import generate_project
from datasci.core.helper import generate_experiment
from datasci.core.dataset import load_dataset
import shutil

# Create a project directory structure in the test path
file_path = './test_data/'
generate_project('Iris', file_path)

# move data into Data directory of Iris project directory
shutil.move('./test_data/iris.ds', './test_data/Iris/Data/iris.ds')

# create experiment directory in the Experiments directory of the Iris directory
proj_dir = './test_data/Iris/'
generate_experiment('setosa_versicolor_classify_species_svm', proj_dir)
```
Once the *setosa_versicolor_classify_species_svm* directory is created there will be a file *setosa_versicolor_classify_species_svm_params.py* containing a template for experimental parameters that the user can change or add on to. The Scripts directory in the Iris directory should contain general purpose scripts that can take in specific experimental parameters from your different experiments—allowing you to easily change your experiment on the fly with minimal code change. Take a look at the [Iris](test_data/Iris) directory for an example of this workflow.

