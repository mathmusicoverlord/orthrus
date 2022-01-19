#!/bin/bash
cd c:/Users/ekeho/Documents/orthrus/docsrc
source c:/Users/ekeho/anaconda3/etc/profile.d/conda.sh
conda activate orthrus
sphinx-apidoc --implicit-namespaces -o rst ../orthrus/
make github
