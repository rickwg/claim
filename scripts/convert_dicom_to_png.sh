#!/usr/bin/env bash
#SBATCH --job claim
#SBATCH --partition=cpu-test
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=3G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/claim-job-%j.out

# $1 PROJECT_DIR
# $2 data directory DATA_DIR
PROJECT_DIR=$1
DATA_DIR=$2

cd $PROJECT_DIR
ls -l
apptainer run \
    --env "FILE_PATH_TO_PREPROCESS_CONFIG=config/preprocess_config_mame.yaml" \
    --bind "$DATA_DIR:/mnt/vindrmammo" \
    --bind "$PROJECT_DIR:/workdir" \
    $PROJECT_DIR/apptainerfile.sif run_data_preprocessing.py
