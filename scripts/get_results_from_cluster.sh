#!/bin/bash
# Script: get_results_from_cluster.sh

# Description: This script copies the results from the cluster to the local machine.

# Arguments:
# $1: The name of the folder containing the results on the cluster. e.g. nlp-benchmark-2024-01-30-21-38-15
# $2: [Optional] The path to the artifacts folder on the cluster. defaults to $HYDRA_PROJECT_DIR/artifacts

# if dot env file exists, source it
if [ -f .env ]; then
  source .env
else
  echo "No .env file found. Please create one."
  exit 1
fi

# If no argument is passed, exit
if [ -z "$1" ]; then
  echo "Experiment name are required"
  exit 1
fi

if [ -z "$2" ]; then
  ARTIFACTS_DIR=$HYDRA_PROJECT_DIR/artifacts
else
  ARTIFACTS_DIR=$2
fi

HYDRA_SSH="$HYDRA_SSH_USER@${HYDRA_HOST:-hydra}"

mkdir -p artifacts/$1

rsync -rv --exclude='models' $HYDRA_SSH:$ARTIFACTS_DIR/"$1"/training artifacts/$1/
#rsync -rv --exclude='*.png' $HYDRA_SSH:$ARTIFACTS_DIR/"$1"/xai artifacts/$1/
rsync -rv --partial --info=progress2 --exclude='intermediate_*' $HYDRA_SSH:$ARTIFACTS_DIR/"$1"/xai_evaluation artifacts/$1/
#rsync -rv $HYDRA_SSH:$ARTIFACTS_DIR/"$1"/analyses artifacts/$1/
#rsync -v $HYDRA_SSH:$ARTIFACTS_DIR/"$1"/experiment_config.json artifacts/$1/
#rsync -v $HYDRA_SSH:$ARTIFACTS_DIR/"$1"/training_split_metadata.jsonl artifacts/$1/
#rsync -rv $HYDRA_SSH:$ARTIFACTS_DIR/"$1"/cluster artifacts/$1/

echo "Results copied to artifacts/$1"


