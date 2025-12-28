#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/works/autogluon"
export nnUNet_raw="${nnUNet_raw:-$ROOT/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$ROOT/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$ROOT/nnUNet_results}"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"

