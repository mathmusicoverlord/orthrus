#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate DataSci
sphinx-apidoc -o rst ../datasci/*
sphinx-apidoc -o rst ../datasci/sparse/*
make github
