# Package Cleaning Check

The released archive was cleaned for review.

Included:

- Source code
- Requirements and environment files
- Local-path examples that use relative placeholders
- Documentation for expected local file layouts
- Qwen2-VL-7B helper entrypoint and 7B-specific NLVR2/ScienceQA loaders
- Tri-modal CMU-MOSI and CMU-MOSEI controlled late-fusion entrypoint

Excluded:

- Datasets
- Pretrained checkpoints
- Trained checkpoints
- Logs and predictions
- Private absolute paths
- User names, email addresses, machine names, and external links
- Non-English or Chinese text in file names or source files

Checks performed before packaging:

- File names contain no Chinese characters.
- Text files contain no Chinese characters.
- Text files contain no external links.
- Text files contain no email addresses.
- Text files contain no user-specific identifiers from the local workspace.
- Text files contain no private absolute machine paths.
- The archive contains no `__pycache__`, bytecode files, checkpoints, logs, or prediction artifacts.
- All Python source files pass syntax compilation.

Additional notes:

- All data and model examples use relative placeholder paths.
- The default `--local_files_only true` behavior is used in the documented examples.
