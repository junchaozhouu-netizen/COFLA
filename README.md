# COFLA AAAI-27 Reproducibility Package

This archive contains the anonymous code package for the paper submission. It is intended for reviewer-side reproduction of the main COFLA experiments without including datasets, pretrained checkpoints, trained weights, private paths, or external download links.

## Contents

```text
COFLA_AAAI2027_Code/
  datasets/                              Dataset loaders and collators
  methods/                               Vanilla, SAM-family, DGL, COFLA, and FAST-COFLA training logic
  models/                                Vision-language and controlled late-fusion model wrappers
  utils/                                 Metrics, seeding, logging, optimizer, and I/O helpers
  train.py                               Full vision-language model entrypoint
  evaluate.py                            Checkpoint evaluation entrypoint
  train_controlled_latefusion.py          Hateful Memes and MM-IMDb controlled late-fusion experiments
  train_controlled_nlvr2_scienceqa.py     NLVR2 and ScienceQA controlled late-fusion experiments
  train_controlled_latefusion_ablation.py Controlled ablation experiments
  train_controlled_latefusion_ablation_sgd.py Controlled SGD breadth-check experiments
  summarize_grid_results.py              Result aggregation utility
  requirements.txt                       Python package list
  environment.yml                        Optional Conda environment specification
  DATA_LAYOUT.md                         Expected local data and checkpoint layout
  ANONYMIZATION_CHECK.md                  Cleaning and verification notes
```

## Environment

A typical setup is:

```bash
conda env create -f environment.yml
conda activate cofla_aaai2027
pip install -r requirements.txt
```

The package assumes that all datasets and pretrained model checkpoints have already been obtained from their official providers and placed in local folders. This archive intentionally does not include download URLs.

## Expected local paths

Default paths are relative to the repository root:

```text
external_models/
  qwen2_5_vl_3b_instruct/
  clip-vit-base-patch32/
  roberta-base/
data/
  hateful_memes/
  nlvr2/
  scienceqa/
  mmimdb/
outputs/
```

You may override every path from the command line. The defaults are placeholders only and contain no private machine paths.

## Full vision-language model training

Example smoke run on a small subset:

```bash
python train.py \
  --method cofla \
  --dataset hateful_memes \
  --model_path ./external_models/qwen2_5_vl_3b_instruct \
  --data_root ./data/hateful_memes \
  --result_root ./outputs/full_vlm \
  --exp_name smoke_hm_cofla \
  --num_train_epochs 1 \
  --max_train_samples 32 \
  --max_val_samples 32 \
  --max_test_samples 32 \
  --local_files_only true
```

For NLVR2, ScienceQA, and MM-IMDb, set `--dataset` and `--data_root` accordingly. Dataset-specific split arguments are available in `train.py`.

## Controlled late-fusion experiments

Hateful Memes or MM-IMDb:

```bash
python train_controlled_latefusion.py \
  --dataset hateful_memes \
  --fusion_type concat_mlp \
  --method cofla \
  --clip_path ./external_models/clip-vit-base-patch32 \
  --roberta_path ./external_models/roberta-base \
  --hm_root ./data/hateful_memes \
  --mmimdb_root ./data/mmimdb \
  --result_root ./outputs/controlled_latefusion \
  --num_train_epochs 1 \
  --max_train_samples 32 \
  --max_val_samples 32 \
  --max_test_samples 32 \
  --local_files_only true
```

NLVR2 or ScienceQA:

```bash
python train_controlled_nlvr2_scienceqa.py \
  --dataset nlvr2 \
  --fusion_type gated_mlp \
  --method cofla \
  --clip_path ./external_models/clip-vit-base-patch32 \
  --roberta_path ./external_models/roberta-base \
  --nlvr2_root ./data/nlvr2 \
  --scienceqa_root ./data/scienceqa \
  --result_root ./outputs/controlled_latefusion \
  --num_train_epochs 1 \
  --max_train_samples 32 \
  --max_val_samples 32 \
  --max_test_samples 32 \
  --local_files_only true
```

## Evaluation

After training with `train.py`, evaluate a saved experiment directory as follows:

```bash
python evaluate.py \
  --exp_dir ./outputs/full_vlm/smoke_hm_cofla \
  --method cofla \
  --dataset hateful_memes \
  --split test \
  --checkpoint_type best
```

## Result aggregation

```bash
python summarize_grid_results.py \
  --result_dir ./outputs/controlled_latefusion \
  --output_prefix cofla_grid
```

## Reproducibility notes

- This package contains code only. It excludes datasets, pretrained checkpoints, trained checkpoints, logs, and predictions.
- Random seeds are exposed through command-line arguments.
- The default loader mode is `--local_files_only true` to avoid implicit network access.
- Any path to data or models should be supplied by the reviewer according to their local environment.
