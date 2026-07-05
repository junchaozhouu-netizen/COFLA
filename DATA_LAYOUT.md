# Expected Local Data and Checkpoint Layout

This package does not include datasets or pretrained checkpoints. Place local files under the repository root or pass custom paths from the command line.

## Hateful Memes

Default root: `./data/hateful_memes`

Expected files:

```text
train.jsonl
dev_seen.jsonl
test_seen.jsonl
img/
```

Common record fields include an image path field, a text field, a binary label, and an optional sample id.

## NLVR2

Default root: `./data/nlvr2`

The standard controlled loader searches common JSONL or Parquet split names for train, validation, and test. The 7B-specific loader expects Parquet files and searches both the root folder and a nested `data/` folder.

Common 7B Parquet split patterns:

```text
balanced_train*.parquet
balanced_dev*.parquet
balanced_test_public*.parquet
balanced_test_unseen*.parquet
unbalanced_train*.parquet
unbalanced_dev*.parquet
unbalanced_test_public*.parquet
unbalanced_test_unseen*.parquet
```

Accepted NLVR2 row fields include:

```text
sentence, statement, question
label, answer, gold_label
left_image, image_left, left, image1, left_img
right_image, image_right, right, image2, right_img
identifier, id, uid, sample_id
```

## ScienceQA

Default root: `./data/scienceqa`

The standard controlled loader searches common JSONL or Parquet split names for train, validation, and test. The 7B-specific loader expects Parquet files and searches both the root folder and a nested `data/` folder.

Common 7B Parquet split patterns:

```text
train*.parquet
validation*.parquet
test*.parquet
```

Accepted ScienceQA row fields include:

```text
question, query, problem
choices, options
answer, label, target
image, image_path, image_file, picture
hint, context
id, problem_id, pid, sample_id
```

## MM-IMDb

Default root: `./data/mmimdb`

Expected files can be JSONL or Parquet, with text and image fields plus multi-label targets. The exact label list can be inferred from the dataset files or supplied through command-line arguments.

## CMU-MOSI and CMU-MOSEI

Default roots: `./data/mosi` and `./data/mosei`

Expected file:

```text
aligned_50.pkl
```

The pickle file should contain `train`, `valid`, and `test` splits. Each split should contain these arrays:

```text
text                shape: N x 50 x D_text
audio               shape: N x 50 x D_audio
vision              shape: N x 50 x D_vision
regression_labels   shape: N or N x 1
```

Optional fields such as `id` and `raw_text` are allowed but not required.

## Pretrained checkpoints

Default roots:

```text
./external_models/qwen2_5_vl_3b_instruct
./external_models/qwen2_vl_7b_instruct
./external_models/clip-vit-base-patch32
./external_models/roberta-base
```

All checkpoint paths can be overridden from the command line.
