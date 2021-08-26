#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate orthrus
sphinx-apidoc --implicit-namespaces -o rst ../orthrus/
make github
