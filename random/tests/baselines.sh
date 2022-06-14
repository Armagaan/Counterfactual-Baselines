#!/usr/bin/bash

# Activate the conda environment.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cfsqr

echo "<----- BA-SHAPES ----->"
python tests/tests.py bashapes
echo

echo "<----- TREE-CYCLES ----->"
python tests/tests.py treecycles
echo

echo "<----- TREE-GRIDS ----->"
python tests/tests.py treecycles
echo