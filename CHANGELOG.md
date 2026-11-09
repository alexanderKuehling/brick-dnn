# Changelog

## Unreleased

### Added
- **Classical models: Gaussian processes and support vector machines.** New bricks next to
  the neural layers: kernels (`RBF`, `Matern`, `RationalQuadratic`, `ExpSineSquared`,
  `DotProduct`, `WhiteKernel`, `ConstantKernel`, combined with `KernelSum` and
  `KernelProduct`) and classical models (`GaussianProcessRegressor`,
  `GaussianProcessClassifier`, `SVR`, `SVC`). Connections carry a tensor or a kernel; the
  analysis checks the rules (a model takes its features straight from the Input, one model per
  network, no mixing with neural layers yet) and reports kernel expressions and the
  hyperparameters the fit optimises. The export is a scikit-learn script (`make_model()`
  returns `Pipeline(StandardScaler(), model)`; CSV fitting with metrics, `--save model.joblib`,
  `--predict`). They train on CSV files, prepared datasets and demo data with scikit-learn,
  fitted once instead of epoch by epoch, at most `BRICKDNN_GP_MAX_ROWS` (5,000) training rows
  for Gaussian processes and `BRICKDNN_SVM_MAX_ROWS` (50,000) for SVMs; runs report the fitted
  kernel and log marginal likelihood or the support vectors, save the fitted pipeline as a
  `.joblib` file, and predict class probabilities or values with the Gaussian process's
  uncertainty (REST, MCP and `brickdnn train`). Examples: Gaussian process regression, SVM
  classifier, periodic Gaussian process. The catalogue (`GET /api/layers/`) tells each brick's
  `family`, `output_kind` and `input_slots`; the analysis adds `output_kind`, `kernel` and
  `task` per node and `family` and `classical` for the network; `GET /api/training/` adds
  `classical`, runs `family`, and the run metrics `val_r2` and `test_r2`.
- **Privacy basics for public servers** ([docs/privacy.md](docs/privacy.md)).
  - *Legal pages:* an imprint, privacy policy and terms of use are Markdown files the
    operator puts into `BRICKDNN_LEGAL_DIR` (on the server `/srv/brick-dnn/legal`), shown at
    `/imprint`, `/privacy` and `/terms` with a safe Markdown subset (no HTML) and linked from
    the header, the phone menu and the login and sign-up pages when they exist. German
    templates and a checklist of what the privacy policy must cover are in `deploy/legal/`.
  - *Terms at sign-up:* with terms of use, sign-up asks to accept them; the acceptance is
    stored with the versions of the terms and the privacy policy.
  - *Account page* (`/account`): change the e-mail address and the password, download
    everything stored about the account as a ZIP file (JSON and the uploaded files), delete
    the account with everything it owns (password and username to confirm; running trainings
    are stopped, files removed, logins ended, live editors told).
  - *Automatic deletion (off by default):* `BRICKDNN_DELETE_UNUSED_UPLOADS_DAYS` deletes data
    sources, datasets and finished runs nobody has used for that long,
    `BRICKDNN_DELETE_INACTIVE_ACCOUNTS_DAYS` inactive accounts. `manage.py purge_old_data
    [--dry-run]` does it; a small `cleanup` service in Docker Compose runs it daily and also
    removes expired logins.
  - Uploads in the Data tab and for training ask not to upload personal or confidential data.
  - Docker keeps at most 3 × 10 MB of logs per container; nginx logs shortened IP addresses
    and paths without query strings; the app is sent with `Referrer-Policy`,
    `X-Content-Type-Options` and `Permissions-Policy` headers.
- The Input layer follows the training data: choosing a dataset (also with *Train with this
  data* in the Data tab) sets the Input shape and the number of outputs when only those keep
  the network from fitting, with a notification and *Undo*; an empty canvas gets a small
  network for the data. Otherwise *Adapt network to “…”* in the training panel and *Set from
  dataset* in the Input layer's properties do it with one click. `POST /api/training/check/`
  (and the MCP tool `check_training`) return these changes as `fixes`. Image folders also fit
  a flat Input (`3·S·S` or `S·S` values) now, like the built-in datasets.
- **Sanity checks in the Data tab's Readiness view.** Besides what blocks the export, the
  view now finds what makes a network poor or its scores misleading, each with the affected
  columns and, where a step solves it, a one-click fix: possible target leakage (a feature
  that is a copy of the target, correlates almost perfectly with it, determines the class by
  its values or by one threshold, or is named like `<target>_label`), validation/test rows
  that are copies of training rows, duplicate rows, rows with the same features but
  different classes, ID columns, near-constant and redundant features, strongly skewed
  features and targets, very different scales, rare classes and classes missing from the
  validation or test rows, and too few training rows for the number of features. Category
  columns get a more precise error: numbers or dates stored as text (convert them),
  spellings that differ only in case or spaces (unify them), a value per row (drop) or many
  categories (frequency or target mean encoding instead of one-hot). The checks learn from
  the training rows (a sample on large tables) and are grouped into problems, warnings and
  suggestions.
- Data tab: *Change type* to numbers can ignore units, currency signs and thousands
  separators (`$1,200`, `12 kg`, `45 %`).
- Data tab: the target column setting points out how to get a target from another table and
  adds a *Join another table* step in one click (with a single other table, it is chosen and a
  key column both tables share, such as `id`, is set).
- **Sharing networks.** Owners invite other users by username or e-mail as editors or
  viewers, or share a view link (`/share/<token>`) that works without an account; members
  see shared networks next to their own and can leave, anyone can make a copy. Saving checks
  the version, so nobody overwrites someone else's changes unnoticed (`409` with the current
  state). The MCP tools list shared networks and respect the roles.
- **Live editing.** Everyone with access works on a network at the same time over a WebSocket
  (`/api/ws/projects/<id>/`, Django Channels): changes are applied on the server in order and
  reach everyone at once, and you see who else is here and what they select. In the builder a
  saved network saves itself as you edit (a status replaces the save button; changes made
  while offline are sent when the connection is back), and the avatars, selection outlines
  and layers being dragged show who does what. Docker Compose runs a small realtime service
  and Redis for it; `manage.py runserver` needs neither.
- **Version history and comments.** Every change is kept (merged per person and session);
  name versions, preview any of them on a read-only canvas and restore it. Comment on a
  network or a layer, reply and resolve threads, viewers too. In the builder, *History* and
  *Comments* open a side panel that follows the live connection, and layers show how many
  open comments they have.
- **Shared runs and data.** Trainings started from a shared network are visible to all its
  members; datasets and data preparations can be shared in a network, so members train on
  them and editors improve the preparation.
- **Predictions with trained networks.** `POST /api/training/runs/<id>/predict/` and the MCP
  tool `predict` run a finished run's network on up to 64 samples on the server and return
  the most likely classes with their probabilities, or the predicted values in the units of
  the target. Inputs are prepared like the training data (CSV rows by column name or in
  column order; images as arrays with pixels 0–255, 0–1 or already normalised). The network
  runs in a separate process with a timeout, one prediction at a time by default
  (`BRICKDNN_MAX_PREDICTIONS`), with its own rate limit (`BRICKDNN_PREDICT_RATE`).
- **More MCP tools.** `get_weights_download` gives assistants a download link for the trained
  weights, valid for 15 minutes without login (signed for the run and the user, and refused
  once the user loses access), together with the model code, the preprocessing and a snippet
  that loads the weights. `upload_dataset` stores a small CSV file sent as text as a training
  dataset (up to `BRICKDNN_MCP_UPLOAD_MB`, default 5 MB), optionally shared in a project.
  `delete_project` (owner), `delete_training_run` (who started it, the project's owner) and
  `delete_dataset` (own datasets not in use) are marked destructive; read-only access never
  sees them. The REST API and MCP use the same delete code, so live editors are told when a
  project is deleted either way.
- **The preprocessing is saved with the weights.** A run's weights come with
  `weights.preprocessing.json` (normalisation statistics of the training data, image size or
  CSV columns, class names), the run view offers it as *Preprocessing (.json)* next to the
  weights, and `brickdnn train` writes `<name>.preprocessing.json` next to `<name>.pt`.
- **Predictions with the generated CSV script**: `python model.py --predict new.csv` loads the
  saved model and prints a prediction for every row, prepared like the training data.
- **Early stopping**: `patience` (in the editor under *Advanced settings*, `--patience` on the
  command line) ends a training when the validation loss has not improved for that many
  epochs. Off by default.
- **Upload progress.** Importing a CSV file in the Data tab and uploading a training dataset
  show a progress bar with the percentage and the megabytes sent, then a striped bar while the
  server reads and checks the file. A training upload can be cancelled while it is sent.

### Changed
- The backend requires **scikit-learn** (for classical models; it brings SciPy and joblib),
  which makes the Docker image about 190 MB larger. `brickdnn[train]` includes it, and the new
  extra `brickdnn[classic]` installs only it.
- **Built-in datasets validate on their training images.** Without a test set, MNIST,
  Fashion-MNIST and CIFAR-10 used to validate on their official test images and ignore
  `val_split`, so the images that chose the best epoch were also the reported result. Now
  `val_split` of the training images is always the validation data, and the official test
  images are always the test set, measured once after training; `test_split` is not used for
  them. `GET /api/training/` names the number of official test images `test_size` (was
  `val_size`).
- **Training keeps the weights of the best epoch.** The weights with the lowest validation
  loss are now tested and saved (`weights.pt`) instead of those of the last epoch, also when a
  run is stopped; `restore_best: false` (`--no-restore-best`) keeps the last epoch as before.
  The run state names the epoch (`weights_epoch`) and whether the run `stopped_early`, the
  run's `metrics` are those of that epoch once it has ended, and the result message says so,
  e.g. *"Stopped early after epoch 12: no improvement for 5 epochs; kept the weights of epoch
  7."* A hyperparameter search does the same per trial and ranks the trials by the validation
  loss of their kept weights instead of their last epoch.
- **Generated CSV training scripts** standardise the features with the mean and standard
  deviation of the training rows (they used all rows, including the validation rows), split
  the rows with a seed (`--seed`, default 0, also for the initial weights) and save a
  checkpoint with the weights and the preprocessing: `torch.load("model.pt")` now returns
  `{"state_dict": ..., "preprocessing": ...}` instead of the bare `state_dict`.
- An e-mail address is required to create an account, and each address can be used by one
  account only (people are invited to shared networks by e-mail, too). Existing accounts
  without one keep working.

### Fixed
- The analysis (`POST /api/analyze/`, MCP `analyze_network`) leaves out parameters that have
  no effect with the other settings, such as `padding_size` when `padding` is `same` or
  `valid`, and lists the resolved `inputs` and `outputs` also for graphs with errors.

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
