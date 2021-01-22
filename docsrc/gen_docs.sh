#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate DataSci

global() {
  shopt -s globstar
  origdir="../datasci/"
  for i in **/; do
    sphinx-apidoc -o rst i
  done
}

sphinx-apidoc -o rst ../datasci/*
make github
