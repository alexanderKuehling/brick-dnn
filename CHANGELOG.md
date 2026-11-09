# Changelog

## Unreleased

### Added
- **Sharing networks.** Owners invite other users by username or e-mail as editors or
  viewers, or share a view link (`/share/<token>`) that works without an account; members
  see shared networks next to their own and can leave, anyone can make a copy. Saving checks
  the version, so nobody overwrites someone else's changes unnoticed (`409` with the current
  state). The MCP tools list shared networks and respect the roles.
- **Live editing.** Everyone with access works on a network at the same time over a WebSocket
  (`/api/ws/projects/<id>/`, Django Channels): changes are applied on the server in order and
  reach everyone at once, and you see who else is here and what they select. Docker Compose
  runs a small realtime service and Redis for it; `manage.py runserver` needs neither.
- **Version history and comments.** Every change is kept (merged per person and session);
  name versions and restore any of them. Comment on a network or a layer, reply and resolve
  threads, viewers too.
- **Shared runs and data.** Trainings started from a shared network are visible to all its
  members; datasets and data preparations can be shared in a network, so members train on
  them and editors improve the preparation.

## 1.1.0 (2026-11-09)

Highlights: connect AI assistants such as Claude to Brick DNN (MCP server with OAuth sign-in
and API tokens), a *Data* tab for preparing and analysing CSV data, training with test sets
and hyperparameter search, PostgreSQL/Supabase, one-command server deployment, an online
demo without a server, and an editor that works on phones.

### Added
- **API tokens and an MCP server.** Create personal API tokens (full access or read only,
  optional expiry, revocable, shown once and stored only as a hash) on the new *API tokens*
  page, and use them for scripts or to connect AI assistants: `/api/mcp/` is a remote MCP
  server whose tools list layers and examples, analyse networks, generate PyTorch code,
  save networks and start, follow and stop trainings as the token's owner. See `docs/mcp.md`.
- **OAuth sign-in for the MCP server**, so it can be added as a custom connector in the
  Claude apps: protected resource and authorization server metadata, dynamic client
  registration, a consent page in the web app (full or read-only access), PKCE, short-lived
  access tokens only valid for MCP, rotating refresh tokens and revocation. Connected apps
  are listed on the *API tokens* page and can be disconnected there.
- **Phones and tablets.** Every page works on phones down to 320px: a page menu in the header,
  finger-sized buttons and handles, a full-screen code view, Data tab and training views that
  fit, and safe areas for the iPhone notch and home indicator. Pinching zooms the network,
  not the page, and the page no longer zooms in by itself when a field gets focus or on a
  double tap (iOS).
- Layers can be dragged out of the palette onto the canvas with the mouse and on touch
  screens (pull sideways or long-press; the palette drawer closes on phones).
- **PostgreSQL and Supabase.** Set `DATABASE_URL=postgresql://…` to keep accounts, networks,
  datasets and runs in PostgreSQL instead of the SQLite file (default). Supabase connection
  strings work as copied from the dashboard: encrypted connections, transaction pooler
  support, and after every `migrate` the tables are hidden from Supabase's Data API (row
  level security, no privileges for `anon`/`authenticated`). Guide, including moving existing
  data, in `docs/database.md`. CI also runs the backend tests on PostgreSQL.
- **Server deployment.** `deploy/server/`: one-time setup script for a small Linux server
  (Docker, firewall, HTTPS with Caddy and Let's Encrypt, a deploy user that can only deploy
  commits of `main`); the *Deploy server* workflow deploys backend and frontend there after
  every successful CI run on `main`. The backend with all its trainings gets at most three
  quarters of the server's CPUs and memory (computed on every deploy), so that Caddy and nginx
  always answer.
- Training processes run with a lower CPU priority than the web server and are the first to
  be killed when memory runs out.
- `manage.py` (`make backend`) and the backend server read settings from `.env` in the
  repository root, like Docker Compose; variables set in the shell win.
- **Data tab: data preparation, integration and analysis.** Import CSV files (separator,
  decimal comma, header, encoding, missing-value placeholders and column types are detected),
  combine tables (join, append) and build a pipeline of steps: missing values (drop, fill
  with mean/median/most frequent/constant/KNN/interpolation), outliers (IQR, z-score, MAD,
  quantiles; clip, remove, set missing, flag), clipping, scaling, log/power/quantile
  transforms, binning, one-hot/ordinal/frequency/target encoding, rare categories, date
  parts, filters, duplicates, formulas, class balancing and feature selection. Every step
  reports what it did; a live preview shows the data, charts (histograms, box plots, missing
  values, correlation heatmaps, scatter plots, target analysis, before/after) and a training
  readiness check with one-click fixes. Values are learned from the training rows only.
  Results become training datasets with a fixed split, cleaned CSV files, JSON recipes or
  standalone pandas scripts. REST endpoints under `/api/data/`, guide in
  `docs/data-preparation.md`.
- Training uses the split of prepared datasets (`DatasetInfo.predefined_split`).
- **Online demo without a server.** `npm run build:demo` builds the editor as static files: the
  code generator runs in the browser with Pyodide, accounts and training are switched off.
  `deploy/huggingface/deploy.sh` publishes it as a free static Hugging Face Space; the
  *Deploy demo* workflow does this after every successful CI run on `main`.
- `brickdnn.service`: the public endpoints (layers, examples, analyze, generate) as plain
  functions, shared by the Django API and the online demo.
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
- The backend depends on pandas and numpy (data preparation); the `brickdnn` code generator
  itself still has no dependencies (`brickdnn[data]` adds them).
- **License changed from MIT to Apache-2.0.** Version 1.0.0 remains available under MIT.
- Frontend upgraded to React 19, Mantine 9, Vite 8 and Vitest 5. Node.js 22.22.2+ (or 24.15+)
  is now required for development.
- Updated backend dependencies (Django 5.2.18, Django REST framework 3.18) and CI actions.
- Dependabot groups updates per ecosystem.
- CI tests the backend on Python 3.12 and 3.13 (3.10 is no longer tested).
- GitHub releases are created from this changelog when a version tag is pushed.

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
