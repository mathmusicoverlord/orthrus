#!/usr/bin/
cd f:/DataSci/docsrc
conda activate DataSci
sphinx-apidoc --implicit-namespaces -o rst ../datasci/
make github
