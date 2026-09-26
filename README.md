# CoTS

Instance-level test-time prompt tuning with TPT, CTPT, OTPT, ATPT, and SOC.

## Setup

```bash
pip install -r requirements.txt
```

The default dataset root is `data_root`. Replace it with your dataset directory
using `--data` or `DATA_ROOT`. CLIP weights are loaded from
`your_cache_path/clip`.

## Single-dataset evaluation

```bash
python instance_tta.py \
  --data /path/to/datasets \
  --cache_dir /path/to/feature_cache \
  --test_sets DTD \
  --algorithm tpt \
  --gpu 0 \
  --arch ViT-B/16 \
  --seed 0
```

The program uses cached image features automatically when it finds
`<cache_dir>/CLIP_seed<seed>/<backbone>/image_<dataset>.pt`. Otherwise, it
loads and augments the original images.

## All datasets

```bash
DATA_ROOT=/path/to/datasets \
CACHE_ROOT=/path/to/feature_cache \
bash run_baseline.sh 0 tpt none
```

Arguments are GPU ID, method, and logit-calibration mode. `DATA_ROOT`,
`CACHE_ROOT`, and `OUTPUT_ROOT` can override the corresponding directories.
