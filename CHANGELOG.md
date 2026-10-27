# Changelog

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
