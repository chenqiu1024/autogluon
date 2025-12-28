#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/isic2017_env.sh"

DATASET="Dataset701_ISIC2017"
CONFIGS="${1:-2d}"

nnUNetv2_find_best_configuration "${DATASET}" -c ${CONFIGS}

