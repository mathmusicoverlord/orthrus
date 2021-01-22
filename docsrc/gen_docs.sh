#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate DataSci
sphinx-apidoc -o rst ../datasci/* ../datasci/sparse/*
make github
