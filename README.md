# COFLA Reproducibility Package

This archive contains a review-ready code package for COFLA experiments. It includes code only and excludes datasets, pretrained checkpoints, trained weights, logs, predictions, private paths, and external links.

## Contents

```text
COFLA_AAAI2027_Code/
  datasets/                                Dataset loaders and collators
  methods/                                 Vanilla, SAM-family, DGL, COFLA, and FAST-COFLA training logic
  models/                                  Vision-language and controlled late-fusion model wrappers
  utils/                                   Metrics, seeding, logging, optimizer, and I/O helpers
  train.py                                 Full vision-language model entrypoint
  train_7B.py                              Qwen2-VL-7B memory-aware entrypoint
  inspect_qwen2vl7b_scope.py               Qwen2-VL-7B trainable-scope inspection utility
  run_commands_sequential_py37.py          Sequential command runner for experiment scripts
  train_controlled_latefusion.py           Hateful Memes and MM-IMDb controlled late-fusion experiments
  train_controlled_nlvr2_scienceqa.py      NLVR2 and ScienceQA controlled late-fusion experiments
  train_tricontrolled_latefusion_strict_cofla.py  Tri-modal CMU-MOSI/CMU-MOSEI controlled experiments
  train_controlled_latefusion_ablation.py  Controlled ablation experiments
  train_controlled_latefusion_ablation_sgd.py  Controlled SGD breadth-check experiments
  summarize_grid_results.py                Result aggregation utility
  requirements.txt                         Python package list
  environment.yml                          Optional Conda environment specification
  DATA_LAYOUT.md                           Expected local data and checkpoint layout
  ANONYMIZATION_CHECK.md                    Package cleaning and verification notes
```

## Environment

A typical setup is:

```bash
conda env create -f environment.yml
conda activate cofla_aaai2027
pip install -r requirements.txt
```

The package assumes that all datasets and pretrained model checkpoints have already been obtained from their original providers and placed in local folders. This archive intentionally does not include any download addresses.

## Expected local paths

Default paths are relative to the repository root:

```text
external_models/
  qwen2_5_vl_3b_instruct/
  qwen2_vl_7b_instruct/
  clip-vit-base-patch32/
  roberta-base/
data/
  hateful_memes/
  nlvr2/
  scienceqa/
  mmimdb/
  mosi/
  mosei/
outputs/
```

Every data or model path can be overridden from the command line. The defaults are placeholders only and contain no private machine paths. See `DATA_LAYOUT.md` for the expected file names and fields.

## Supported method names

Full vision-language model entrypoints support:

```text
vanilla_ft, vanilla_lora, sam_lora, esam_lora, msam_lora, masam_lora, dgl_lora, cofla, fast_cofla
```

Controlled late-fusion entrypoints support:

```text
vanilla, sam, cofla, vanilla_lora, sam_lora, esam_lora, msam_lora, masam_lora, dgl_lora
```

Tri-modal controlled entrypoint supports:

```text
vanilla, sam, esam, msam, masam, dgl, cofla_e, cofla_f
```

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

## Qwen2-VL-7B training

The 7B entrypoint patches model aliases and 7B-specific NLVR2/ScienceQA dataset classes at runtime without modifying the original training entrypoint. It uses conservative memory-aware defaults when a flag is omitted.

```bash
python train_7B.py \
  --method cofla \
  --dataset nlvr2 \
  --model_path ./external_models/qwen2_vl_7b_instruct \
  --data_root ./data/nlvr2 \
  --result_root ./outputs/full_vlm_7b \
  --exp_name smoke_nlvr2_7b_cofla \
  --num_train_epochs 1 \
  --max_train_samples 16 \
  --max_val_samples 16 \
  --max_test_samples 16 \
  --local_files_only true
```

The 7B helper sets default 4-bit quantization, fp16 compute, gradient checkpointing, batch size 1, and gradient accumulation 8 unless these options are explicitly overridden.

To inspect the matched-LoRA trainable scope for the 7B run:

```bash
python inspect_qwen2vl7b_scope.py \
  --model_path ./external_models/qwen2_vl_7b_instruct \
  --out ./outputs/qwen2vl7b_scope_train7B.txt
```

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

## Tri-modal controlled experiments

The tri-modal script supports CMU-MOSI and CMU-MOSEI processed `aligned_50.pkl` features with `text`, `audio`, `vision`, and `regression_labels` fields. The script is self-contained and does not import the full vision-language model code.

```bash
python train_tricontrolled_latefusion_strict_cofla.py \
  --dataset mosi \
  --data_root ./data/mosi \
  --result_root ./outputs/tricontrolled_latefusion \
  --exp_name smoke_mosi_cofla_f \
  --method cofla_f \
  --fusion_type gated_mlp \
  --num_train_epochs 1
```

For CMU-MOSEI, replace `--dataset mosi` and `--data_root ./data/mosi` with `--dataset mosei` and `--data_root ./data/mosei`.

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

## Reproducibility and release notes

- This package contains code only. It excludes datasets, pretrained checkpoints, trained checkpoints, logs, and predictions.
- Random seeds are exposed through command-line arguments.
- The default loader mode is `--local_files_only true` to avoid implicit network access.
- Any path to data or models should be supplied according to the local runtime environment.
- The archive has been checked to avoid Chinese text, external links, email addresses, private absolute paths, and user-specific identifiers.
