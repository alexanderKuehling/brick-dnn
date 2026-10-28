# Changelog

## 1.1.0 (unreleased)

### Added
- **Local training.** A *Train* button in the editor trains the current network on the machine
  that runs the backend (CUDA, Apple GPU or CPU) and shows loss and accuracy live. Runs can be
  stopped, are kept in a history together with the trained network, and the weights can be
  downloaded with the matching script.
- Data sources: demo data with a learnable hidden rule (works for every network), MNIST,
  Fashion-MNIST and CIFAR-10 (downloaded on first use), CSV uploads (classification or
  regression, `,`/`;` separators, text labels) and ZIP uploads with one folder of images per class.
- Compatibility check before training that explains what to change, e.g. the number of outputs
  or the Input shape. Images are converted and resized to the Input layer automatically.
- **Test set.** Optionally hold out a share of the data as a test set (`test_split`). It is
  never used for training or for choosing settings; the final model is measured on it once
  after training. MNIST, Fashion-MNIST and CIFAR-10 use their official test images for it and
  validate on part of their training images.
- **Hyperparameter search.** Random or grid search over learning rate, batch size, optimizer
  and weight decay. All trials train on the same data split from the same initial weights; the
  trial with the lowest validation loss wins and its weights are kept. Trials whose loss
  explodes or that run out of memory are skipped. Available in the editor, on the command line
  (`--tune`, `--trials`, `--tune-lr 1e-4:1e-2`, …) and via the API (`tuning`).
- `brickdnn train` command and the optional `brickdnn[train]` extra.
- REST endpoints under `/api/training/` (see `docs/api.md`) and a training guide
  (`docs/training.md`).
- Docker: optional PyTorch in the backend image (`BRICKDNN_INSTALL_TORCH=true`).

### Changed
- Frontend upgraded to React 19, Mantine 9, Vite 8 and Vitest 5. Node.js 22.13+ is now
  required for development.
- Updated backend dependencies (Django 5.2.18, Django REST framework 3.18) and CI actions.
- Dependabot groups updates per ecosystem.

## 1.0.0 (2026-10-27)

First public release. The former `brick_dnn_api` (Django) and `brick_dnn_app` (React)
repositories are merged into one project and completed.

### Added
- Graph-based code generation: connections are respected, including skip connections,
  multiple inputs/outputs and merge layers (`Add`, `Concat`).
- Shape inference and parameter counts for every layer, with automatic `in_features`,
  `in_channels`, `num_features`, `d_model` …
- 24 layer types (previously 4), including `Conv1d`, pooling, normalisation, `Embedding`,
  `LSTM`/`GRU`/`RNN`, `TransformerEncoder` and `PositionalEmbedding`.
- Generated scripts with sanity check and optional CSV training loop.
- Live validation with clear error messages in the editor.
- Examples, JSON import/export, automatic layout, dark mode, mobile layout.
- Optional accounts with saved networks (JWT with refresh-token rotation).
- Standalone `brickdnn` Python package with CLI.
- Test suites (backend runs generated models in PyTorch), CI, Docker Compose setup, docs.

### Changed
- Configuration via environment variables; no secrets in the repository.
- Building and exporting no longer require logging in.
