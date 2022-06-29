# Activate the conda environment.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cfsqr

echo "<----- BA-SHAPES ALPHA=0.0 ----->"
python tests/tests.py outputs/bashapes/bashapes-alp_0.0-1656483294
echo

echo "<----- BA-SHAPES ALPHA=0.6 ----->"
python tests/tests.py outputs/bashapes/bashapes-alp_0.6-1656486154
echo

echo "<----- TREE-CYCLES ALPHA=0.0 ----->"
python tests/tests.py outputs/treecycles/treecycles-alp_0.0-1656482998
echo

echo "<----- TREE-CYCLES ALPHA=0.6 ----->"
python tests/tests.py outputs/treecycles/treecycles-alp_0.6-1656483099
echo

echo "<----- TREE-GRIDS ALPHA=0.0 ----->"
python tests/tests.py outputs/treegrids/treegrids-alp_0.0-1656491030
echo

echo "<----- TREE-GRIDS ALPHA=0.6 ----->"
python tests/tests.py outputs/treegrids/treegrids-alp_0.6-1656491434
echo