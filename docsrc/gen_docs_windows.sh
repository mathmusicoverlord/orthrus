#!/bin/bash
cd c:/Users/ekeho/Documents/orthrus/docsrc
conda activate orthrus
sphinx-apidoc --implicit-namespaces -o rst ../orthrus/
make github
