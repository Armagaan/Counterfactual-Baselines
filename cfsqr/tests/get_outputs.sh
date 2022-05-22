# * Run this script prior to running the notebooks.
# * The output of this script is needed in those notebooks.

# ! Run this script from the root directory of the repo.

usage() {
    # echo correct usage to cosole
    echo
    echo "USAGE: bash ${0} [-d DATASET] [-a ALPHA]" >&2
    echo
    echo "DATASET:  path to dataset"
    echo "ALPHA:    value in [0,1]"
    exit 1
}

# Check whether the required number of arguments are supplied.
if [ ${#} != 4 ]; then
    usage
fi

# Parse command line arguments
while getopts "a:d:" OPTION; do
    case ${OPTION} in
        d) DATASET=${OPTARG} ;;
        a) ALP=${OPTARG} ;;
        ?) usage ;;
    esac
done
shift "$(( OPTIND - 1 ))"

# Add present directory to python path. This is required by the authors.
source setup.sh

# Activate the conda environment.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cfsqr

# Choose a script based on the supplied dataset.
case ${DATASET} in
    bashapes)
        python scripts/exp_node_ba_shapes.py --alp="$ALP" > outputs/bashapes/log-"$DATASET"-"$ALP".txt
        ;;
    treecycles)
        python scripts/exp_node_tree_cycles.py --alp="$ALP" > outputs/treecycles/log-"$DATASET"-"$ALP".txt
        ;;
    treegrids)
        python scripts/exp_node_tree_grids.py --alp="$ALP" > outputs/treegrids/log-"$DATASET"-"$ALP".txt
        ;;
    *)
        echo "Invalid dataset" >&2
        exit 1
        ;;
esac

exit 0
