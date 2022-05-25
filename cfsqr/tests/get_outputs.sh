# * Run this script prior to running the notebooks.
# * The output of this script is needed in those notebooks.

# ! Run this script from the root directory of the repo.

usage() {
    # echo correct usage to cosole
    echo
    echo "USAGE: bash ${0} [-d DATASET] [-a ALPHA]" >&2
    echo
    echo "DATASET:  Dataset name. One of [bashapes, treecycles, treegrids]"
    echo "ALPHA:    Value in [0,1]. Smaller the value, greater the counterfactual behaviour"
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

DATASETS="bashapes treecycles treegrids"
if [[ ! " ${DATASETS[*]} " =~ " ${DATASET} " ]]; then
    echo "Invalid dataset"
    usage
elif [ "$ALP" > 1 ] || [ "$ALP" < 0 ]; then
    echo "Invalid value for alpha!"
    usage
fi

# Add present directory to python path. This is required by the authors.
source setup.sh

# Activate the conda environment.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cfsqr

# Create a folder for storing outputs
NOW=$(date +'%s') # present time in milliseconds

FOLDER="outputs/${DATASET}/${DATASET}-alp_${ALP}-${NOW}"
mkdir "$FOLDER"

# Choose a script based on the supplied dataset.
case ${DATASET} in
    bashapes)
        python scripts/exp_node_ba_shapes.py --alp="$ALP" --output="$FOLDER" > "$FOLDER"/log.txt
        ;;
    treecycles)
        python scripts/exp_node_tree_cycles.py --alp="$ALP" --output="$FOLDER" > "$FOLDER"/log.txt
        ;;
    treegrids)
        python scripts/exp_node_tree_grids.py --alp="$ALP" --output="$FOLDER" > "$FOLDER"/log.txt
        ;;
    *)
        echo "Something's wrong" >&2
        exit 1
        ;;
esac

exit 0
