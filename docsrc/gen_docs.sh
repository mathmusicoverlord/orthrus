#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate DataSci
sphinx-apidoc -o rst ../datasci/* --tocfile core
sphinx-apidoc -o rst ../datasci/sparse/* --tocfile sparse
make github
