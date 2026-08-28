#!/usr/bin/env bash
#SBATCH --job claim
#SBATCH --partition=cpu-5h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
##SBATCH --constraint="80gb|40gb"
#SBATCH --output=logs/claim-job-%j.out

# $1 PROJECT_DIR
# $2 SQFS_FILE (filename of the sqfs dataset inside $SQFS_DIR)
# $3 ARTIFACT_DIR
# $4 CONFIG_PATH (set as experiment and preprocess config env vars)
# $5+ Script and arguments to run inside the container
#   e.g. run_experiments.py --mode training
#   e.g. run_data_preprocessing.py

PROJECT_DIR=$1
SQFS_FILE=$2
ARTIFACT_DIR=$3
CONFIG=$4
shift 4

SQFS_DIR="${SQFS_DIR:?Set SQFS_DIR to the directory holding the .sqfs datasets}"
SQFS_SRC="$SQFS_DIR/$SQFS_FILE"
cp "$SQFS_SRC" /tmp/

cd $PROJECT_DIR
ls -l
apptainer run \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --env "FILE_PATH_TO_EXPERIMENT_CONFIG=$CONFIG" \
    --env "FILE_PATH_TO_PREPROCESS_CONFIG=$CONFIG" \
    --nv \
    --bind "/tmp/$SQFS_FILE:/mnt/data:image-src=/" \
    --bind "$ARTIFACT_DIR:/mnt/artifacts" \
    --bind "$PROJECT_DIR:/workdir" \
    $PROJECT_DIR/apptainerfile.sif "$@"