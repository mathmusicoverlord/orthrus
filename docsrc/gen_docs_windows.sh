#!/usr/bin/
cd c:/Users/ekeho/Documents/DataSci/docsrc
conda activate DataSci
sphinx-apidoc --implicit-namespaces -o rst ../datasci/
make github
