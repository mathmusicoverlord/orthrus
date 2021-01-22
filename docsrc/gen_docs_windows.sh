#!/usr/bin/
cd f:/DataSci/docsrc
conda activate DataSci
sphinx-apidoc -o rst ../datasci/* --tocfile core
sphinx-apidoc -o rst ../datasci/sparse/* --tocfile sparse
make github
