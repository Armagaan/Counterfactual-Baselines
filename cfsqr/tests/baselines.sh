# Activate the conda environment.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cfsqr

echo "<----- BA-SHAPES ALPHA=0.0 ----->"
python tests/tests.py outputs/bashapes/bashapes-alp_0.0-1656513715
echo

echo "<----- BA-SHAPES ALPHA=0.6 ----->"
python tests/tests.py outputs/bashapes/bashapes-alp_0.6-1656516517
echo

echo "<----- TREE-CYCLES ALPHA=0.0 ----->"
python tests/tests.py outputs/treecycles/treecycles-alp_0.0-1656512240
echo

echo "<----- TREE-CYCLES ALPHA=0.6 ----->"
python tests/tests.py outputs/treecycles/treecycles-alp_0.6-1656512328
echo

echo "<----- TREE-GRIDS ALPHA=0.0 ----->"
python tests/tests.py outputs/treegrids/treegrids-alp_0.0-1656511167
echo

echo "<----- TREE-GRIDS ALPHA=0.6 ----->"
python tests/tests.py outputs/treegrids/treegrids-alp_0.6-1656511427
echo