# Activate the conda environment.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cfsqr

echo "<----- BA-SHAPES ALPHA=0.0 ----->"
python tests/tests.py outputs/bashapes/bashapes-alp_0.0-1653373398
echo

echo "<----- BA-SHAPES ALPHA=0.6 ----->"
python tests/tests.py outputs/bashapes/bashapes-alp_0.6-1653373410
echo

echo "<----- TREE-CYCLES ALPHA=0.0 ----->"
python tests/tests.py outputs/treecycles/treecycles-alp_0.0-1653386395
echo

echo "<----- TREE-CYCLES ALPHA=0.6 ----->"
python tests/tests.py outputs/treecycles/treecycles-alp_0.6-1653386374
echo