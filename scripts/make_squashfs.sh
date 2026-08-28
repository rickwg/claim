#!/usr/bin/env bash
#SBATCH --job claim
#SBATCH --partition=cpu-5h
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G per cpu-core is default)
##SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/claim-job-%j.out

# $1 source dataset directory, $2 output .sqfs path
SRC_DIR="${1:?Usage: sbatch make_squashfs.sh <src-dir> <out.sqfs>}"
OUT_SQFS="${2:?Usage: sbatch make_squashfs.sh <src-dir> <out.sqfs>}"

squash-dataset "$SRC_DIR" "$OUT_SQFS"
