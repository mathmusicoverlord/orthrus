#!/bin/bash
eval "$(conda shell.bash hook)"
cd docsrc
conda activate DataSci
sphinx-apidoc -o rst ../datasci/*
make github
