# * Run this script prior to running the notebooks.
# * The output of this script is needed in those notebooks.

# ! Run this script from the root directory of the repo.

usage() {
    # echo correct usage to cosole
    echo
    echo "USAGE: bash ${0} [-d DATASET]" >&2
    echo
    echo "DATASET:  Dataset name. One of [bashapes, treecycles, treegrids]"
    exit 1
}

# Check whether the required number of arguments are supplied.
if [ ${#} != 2 ]; then
    usage
fi

# Parse command line arguments
while getopts "d:" OPTION; do
    case ${OPTION} in
        d) DATASET=${OPTARG} ;;
        ?) usage ;;
    esac
done
shift "$(( OPTIND - 1 ))"

DATASETS="bashapes treecycles treegrids"
if [[ ! " ${DATASETS[*]} " =~ " ${DATASET} " ]]; then
    echo "Invalid dataset"
    usage
fi

# Activate the conda environment.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gnn

# Create a folder for storing outputs
NOW=$(date +'%s') # present time in milliseconds

FOLDER="outputs/${DATASET}/${NOW}"
mkdir "$FOLDER"

# Choose a script based on the supplied dataset.
case ${DATASET} in
    bashapes)
        python src/baseline_random.py \
        --dataset=syn1 \
        --num_epochs=500 \
        > "$FOLDER"/log.txt
        ;;
    treecycles)
        python src/baseline_random.py \
        --dataset=syn4 \
        --num_epochs=500 \
        > "$FOLDER"/log.txt
        ;;
    treegrids)
        python src/baseline_random.py \
        --dataset=syn5 \
        --num_epochs=500 \
        > "$FOLDER"/log.txt
        ;;
    *)
        echo "Something's wrong" >&2
        exit 1
        ;;
esac

exit 0
