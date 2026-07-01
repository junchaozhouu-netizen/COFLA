# Anonymization and Clean-Package Check

The released archive was cleaned for anonymous review.

## Removed from the original working copy

- IDE configuration files.
- Python bytecode caches and compiled files.
- Local document-processing utilities unrelated to the paper experiments.
- Toy plotting code that used simulated data rather than experiment outputs.
- Private absolute paths and machine-specific defaults.
- Non-English comments and non-English runtime text.
- External download links and package-index links.
- File names and help strings containing personal or project-local identifiers.

## Verification performed

The final archive was scanned for:

- URL-like strings.
- Email-like strings.
- common private absolute path patterns.
- Chinese characters.
- Python bytecode files and cache folders.
- personal identifier strings detected in the original package.

The source files were also checked with Python bytecode compilation.
