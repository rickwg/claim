#!/usr/bin/env bash
#SBATCH --job claim
#SBATCH --partition=cpu-test
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/claim-job-%j.out

PROJECT_DIR=$1 # absolute path to the project checkout on the cluster
cd $PROJECT_DIR
ls -l
apptainer build --fakeroot --force /tmp/apptainerfile.sif apptainerfile.def
cp /tmp/apptainerfile.sif $PROJECT_DIR/apptainerfile.sif