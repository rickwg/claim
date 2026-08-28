#!/usr/bin/env bash
#SBATCH --job claim
#SBATCH --partition=cpu-test
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=3G         # memory per cpu-core (4G per cpu-core is default)
##SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/claim-job-%j.out

# $1 PROJECT_DIR
# $2 DATA_DIR
# $3 ARTIFACT_DIR
# $4 CONFIG_PATH (set as experiment and preprocess config env vars)
# $5+ Script and arguments to run inside the container
#   e.g. run_experiments.py --mode training
#   e.g. run_data_preprocessing.py

PROJECT_DIR=$1
DATA_DIR=$2
ARTIFACT_DIR=$3
CONFIG=$4
shift 4

cd $PROJECT_DIR
ls -l
apptainer run \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --env "FILE_PATH_TO_EXPERIMENT_CONFIG=$CONFIG" \
    --env "FILE_PATH_TO_PREPROCESS_CONFIG=$CONFIG" \
    --nv \
    --bind "$DATA_DIR:/mnt/data" \
    --bind "$ARTIFACT_DIR:/mnt/artifacts" \
    --bind "$PROJECT_DIR:/workdir" \
    $PROJECT_DIR/apptainerfile.sif "$@"
