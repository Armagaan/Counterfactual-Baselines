# * Run this script prior to running the notebooks.
# * The output of this script is needed in those notebooks.

# ! Run this script from the root directory of the repo.

usage() {
    # echo correct usage to cosole
    echo
    echo "USAGE: bash ${0} [-d DATASET] [-e EVALMODE]" >&2
    echo
    echo "DATASET:  Dataset name. One of [syn1, syn4, syn5]"
    echo "EVALMODE: Whether to run the scripts for transductive version or the inductive version. One of [train, eval]"
    exit 1
}

# Check whether the required number of arguments are supplied.
if [ ${#} != 3 ]; then
    usage
fi

# Parse command line arguments
while getopts "d:" OPTION; do
    case ${OPTION} in
        d) DATASET=${OPTARG} ;;
        e) EVALMODE=${EVALMODE};;
        ?) usage ;;
    esac
done
shift "$(( OPTIND - 1 ))"

DATASETS="syn1 syn4 syn5"
if [[ ! " ${DATASETS[*]} " =~ " ${DATASET} " ]]; then
    echo "Invalid dataset"
    usage
fi

# Activate the conda environment.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gem

# Choose a script based on the supplied dataset.
case ${DATASET} in
    syn1)
        python src/main_explain.py \
        --dataset=syn1 \
        --lr=0.01 \
        --beta=0.5 \
        --n_momentum=0.9 \
        --optimizer=SGD \
        > "$FOLDER"/log.txt
        ;;
    syn4)
        python src/main_explain.py \
        --dataset=syn4 \
        --lr=0.1 \
        --beta=0.5 \
        --optimizer=SGD \
        > "$FOLDER"/log.txt
        ;;
    syn5)
        python src/main_explain.py \
        --dataset=syn5 \
        --lr=0.1 \
        --beta=0.5 \
        --optimizer=SGD \
        > "$FOLDER"/log.txt
        ;;
    *)
        echo "Something's wrong" >&2
        exit 1
        ;;
esac

exit 0
