#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

gpu=${1:-0}
method=${2:-tpt}
calibration=${3:-none}
data_root=${DATA_ROOT:-data_root}
cache_root=${CACHE_ROOT:-feature_cache}
output_root=${OUTPUT_ROOT:-outputs}

datasets=(
    DTD Flower102 Caltech101 Aircraft Pets UCF101 Cars eurosat
    SUN397 Food101 A V R I K
)

for dataset in "${datasets[@]}"; do
    python instance_tta.py \
        --data "${data_root}" \
        --cache_dir "${cache_root}" \
        --output_dir "${output_root}" \
        --test_sets "${dataset}" \
        --batch-size 64 \
        --ctx_init a_photo_of_a \
        --print-freq 50 \
        --algorithm "${method}" \
        --gpu "${gpu}" \
        --arch ViT-B/16 \
        --seed 0 \
        --logit_calibration "${calibration}"
done
