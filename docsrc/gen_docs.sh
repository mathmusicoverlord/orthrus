#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate DataSci
sphinx-apidoc --implicit-namespaces -o rst ../datasci/*
make github
