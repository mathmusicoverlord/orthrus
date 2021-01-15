#!/usr/bin/
cd f:/DataSci/docsrc
conda activate DataSci
sphinx-apidoc -o rst ../datasci/*
make github
