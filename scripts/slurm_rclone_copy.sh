#!/usr/bin/env bash
#SBATCH --job-name=rclone-copy
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rclone-copy-%j.out
#SBATCH --error=logs/rclone-copy-%j.err

set -euo pipefail

# Apptainer image built from apptainerfile-rclone.def.
IMAGE_PATH="${RCLONE_IMAGE_PATH:?Set RCLONE_IMAGE_PATH to the rclone .sif path}"

# Remote configuration. Supply these via the environment (e.g. a sourced .env
# that is NOT committed) -- never hardcode credentials in this file.
REMOTE_NAME="${REMOTE_NAME:-remote}"
REMOTE_TYPE="${REMOTE_TYPE:-sftp}"
REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST}"
REMOTE_USER="${REMOTE_USER:?Set REMOTE_USER}"
REMOTE_PASS="${REMOTE_PASS:?Set REMOTE_PASS}"
REMOTE_PORT="${REMOTE_PORT:-22}"

SRC_DIR="${1:-${SRC_DIR:-}}"
DST_REMOTE="${2:-${DST_REMOTE:-${REMOTE_NAME}:/target/path}}"
RCLONE_FLAGS="${RCLONE_FLAGS:---progress}"

SRC_DIR="$(realpath "${SRC_DIR}")"

mkdir -p logs

APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
APPTAINER_BIND="${SRC_DIR}:${SRC_DIR}"

read -r -a RCLONE_FLAGS_ARR <<< "${RCLONE_FLAGS}"

RCLONE_CONFIG_DIR="$(mktemp -d)"
OBSCURED_PASS="$("${APPTAINER_BIN}" run --bind "${RCLONE_CONFIG_DIR}:/config" "${IMAGE_PATH}" obscure "${REMOTE_PASS}")"
cat > "${RCLONE_CONFIG_DIR}/rclone.conf" <<EOF
[${REMOTE_NAME}]
type = ${REMOTE_TYPE}
host = ${REMOTE_HOST}
user = ${REMOTE_USER}
pass = ${OBSCURED_PASS}
port = ${REMOTE_PORT}
EOF

"${APPTAINER_BIN}" run --bind "${APPTAINER_BIND},${RCLONE_CONFIG_DIR}:/config" "${IMAGE_PATH}" \
  copy "${RCLONE_FLAGS_ARR[@]}" "${SRC_DIR}" "${DST_REMOTE}"

rm -rf "${RCLONE_CONFIG_DIR}"
n