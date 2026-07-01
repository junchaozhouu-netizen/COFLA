# Data and Checkpoint Layout

This code package does not include datasets, pretrained checkpoints, trained weights, or download links. Place externally obtained resources under the following local directories, or pass custom paths through command-line arguments.

## Hateful Memes

Default root: `./data/hateful_memes`

Expected files are JSONL split files and the referenced image files. The default split names are:

```text
train.jsonl
dev_seen.jsonl
dev_unseen.jsonl
```

The loader expects records containing an image reference, text field, and label field. The accepted key aliases are implemented in `datasets/hateful_memes.py`.

## NLVR2

Default root: `./data/nlvr2`

The loader supports the split names configured by:

```text
--nlvr2_train_split train
--nlvr2_val_split dev
--nlvr2_test_split test_public
```

Each record should provide two image references, a statement, and a binary label.

## ScienceQA

Default root: `./data/scienceqa`

The loader supports the split names configured by:

```text
--scienceqa_train_split train
--scienceqa_val_split validation
--scienceqa_test_split test
```

Each record should provide the question, answer choices, answer index, optional hint, and optional image.

## MM-IMDb

Default root: `./data/mmimdb`

The default split file is:

```text
split.json
```

The loader expects movie poster images, plot or text fields, and multi-label genre annotations. Label names can be inferred from the split metadata or passed through `--mmimdb_label_names`.

## Pretrained models

Default roots:

```text
./external_models/qwen2_5_vl_3b_instruct
./external_models/clip-vit-base-patch32
./external_models/roberta-base
```

The directories should contain locally available model files compatible with the `transformers` loaders used in `models/`.
