# * Run this script prior to running the notebooks.
# * The output of this script is needed in those notebooks.

# ! Run this script from the root directory of the repo.

usage() {
    # echo correct usage to cosole
    echo
    echo "USAGE: bash ${0} [-d DATASET] [-a ALPHA]" >&2
    echo
    echo "DATASET:  Dataset name. One of [mutag, ]"
    echo "ALPHA:    Float value in [0.0, 1.0]. Smaller the value, greater the counterfactual behaviour"
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

DATASETS="muta"
MIN=0.0
MAX=1.0
if [[ ! " ${DATASETS[*]} " =~ " ${DATASET} " ]]; then
    echo "Invalid dataset"
    usage
elif [ 1 -eq "$(echo "${ALP} < ${MIN}" | bc)" ] || [ 1 -eq "$(echo "${ALP} > ${MAX}" | bc)" ]
then
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
    muta)
        python scripts/exp_graph.py --alp="$ALP" --output="$FOLDER" > "$FOLDER"/log.txt
        ;;
    *)
        echo "Something's wrong" >&2
        exit 1
        ;;
esac

exit 0
