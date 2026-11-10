# Changelog

## Unreleased

### Changed
- **Compact MCP answers.** `get_training_options` lists datasets without their columns (new
  `get_dataset` pages through them), `get_training_run` is a summary (`include` adds the
  history, at most 50 epochs, the graph, plan, log, trials and preprocessing),
  `list_training_runs` shows the newest 20 runs one line each (`limit`, `status`,
  `project_id`), `list_layer_types` one line per parameter (`types` for full specs),
  `list_examples` names and descriptions (`id` for a graph), and `check_training` and the
  data tools cap long lists with their totals. A run with 200 epochs went from 71 KB to 1 KB.
- gunicorn's access log leaves out query strings, so signed download and upload links are not
  logged.
- **With Docker Compose, trainings are queued and trained by a worker container** (see
  *Added*). `docker compose up` now starts a `worker` service next to the backend, and a
  start no longer answers `409` because another training is running: the run is `queued`.
  `BRICKDNN_TRAINING_WORKER=false` brings the old behaviour back; without Docker
  (`runserver`, `make backend`) nothing changes. `BRICKDNN_TRAINING_MAX_RUNS` becomes the
  default of the worker's slots. Clients that only know the old run statuses should treat
  `queued` like `starting`.
- **Server deployment** (`deploy/server`): the trainings moved out of the web backend into
  the worker, each with its own limits within the same three quarters of the server (on 2
  CPUs and 4 GB: worker 1.5 CPUs and 1.8 GB, backend 1 CPU and 1 GB, and the backend gets the
  CPU first). Servers that were set up earlier need no new copy of `brickdnn-deploy`; with
  one, the memory limits follow the size of the server. A deploy no longer fails the running
  trainings: they go back into the queue and start again.

### Added
- **Example *U-Net classifier (Fashion-MNIST)*.** The U-Net example with a classification
  head: instead of the 1×1 mask convolution, `AdaptiveAvgPool2d` → `Flatten` → `Linear(10)`.
  Its Input is `[1, 32, 32]` (Fashion-MNIST's 28×28 images are resized, so that three
  halvings fit) and its filters are halved (16/32/64, bottleneck 128, 482k parameters), so
  that it trains on a CPU server. It follows the U-Net in the examples; for this task a plain
  CNN is smaller and faster.
- **A guide for new users.** On the first visit, the builder walks through five short steps
  (the palette, connecting layers, the analysis badge and the properties, code and export,
  training with a built-in dataset) with Mantine's `Tour`: keyboard and screen reader
  friendly, at phone width it points at the phone's buttons, *Skip* or Escape end it. It is
  remembered per browser and opens again with the *?* button of the toolbar or *File* →
  *Guide to the builder*; the online demo shows it too. An empty canvas offers cards to start
  from an example (MLP on MNIST, CNN, U-Net, Gaussian process).
- **Kernel hyperparameters at a bound.** After a Gaussian process fit, Brick DNN finds the
  optimised kernel hyperparameters that ended at one of their bounds (scikit-learn only logs a
  `ConvergenceWarning`) and stores them in `state.classical.at_bounds` with the brick, the
  side and a suggested change. The run's results say what it may mean ("Noise level of
  white_1 ended at its lower bound 1e-05: the model may be overfitting; set custom bounds,
  e.g. ≥ 0.001"), and a button sets the suggested custom bounds on the brick in the builder,
  with *Undo*. The MCP tool `get_training_run` includes the messages in its summary.
- **Backups: `manage.py backup` and `manage.py restore`.** A backup is a folder with the
  database as of one moment (Django's `dumpdata` in one transaction, which loads into any
  PostgreSQL or SQLite; `pg_dump` where installed; SQLite's online backup), the files of the
  data directory as a tarball (only listed above `BRICKDNN_BACKUP_MAX_FILES_MB`) and a
  manifest. Old backups are rotated (7 daily, 4 weekly). Docker Compose's cleanup service runs
  it daily after the clean-up when `BRICKDNN_BACKUP_DIR` is set, `/api/health/` reports a
  failed or old backup, and `restore` puts a backup back (`--database-only`, `--files-only`).
  brickdnn.com backs up to `/data/backups`. `deploy/server/README.md` has a new section
  *Backups*: Hetzner's server backups as offsite copy, Supabase's backups, the restore drill.
- **Uptime monitoring without another service.** A scheduled GitHub Actions workflow
  (`uptime.yml`, every 15 minutes) checks `DEPLOY_URL/api/health/` and the front page, and
  when the site is down opens one issue labelled `outage` (GitHub notifies the owner), comments
  on it while the problem lasts or changes, and closes it when the site is back.
  `/api/health/` now checks the database (`503` without it), the training worker's heartbeat
  and the free disk space (`BRICKDNN_HEALTH_MIN_FREE_MB`), and answers `degraded` when one of
  the latter fails. It names the checks only; the details are for staff accounts and requests
  with the `X-Health-Token` header (`BRICKDNN_HEALTH_TOKEN`). See
  [deploy/server/README.md](deploy/server/README.md#monitoring).
- **Administrators from the settings.** `BRICKDNN_ADMIN_USERS` lists usernames whose accounts
  become staff and superusers when the backend starts (Django admin, no limits by default).
  Nobody can register a listed name; removing a name takes no rights away. A public
  server can set it in `deploy/sites/<domain>/settings.env`, so no shell on the server is
  needed. brickdnn.com names its owner's account.
- **U-Net example and `ConvTranspose2d`.** A new brick for transposed 2-D convolutions
  (learned upsampling: kernel 2 with stride 2 doubles the size; padding and output padding
  as in PyTorch), and the example *U-Net (image segmentation)*: three levels down with
  `MaxPool2d`, three up with `ConvTranspose2d`, skip connections with `Concat`, one value per
  pixel as output. It is an example for building and exporting code: the datasets the app
  trains on have one label per image, not one per pixel.
- **A training queue with a separate worker.** With `BRICKDNN_TRAINING_WORKER` the web server
  only stores a new run as `queued`; `manage.py training_worker` (its own process or
  container, `--slots N`, `--once` for one run) takes the runs from the database and trains
  them with the usual runner, so trainings wait instead of being refused and cannot slow the
  site down. Plans with a higher `priority` start first (accounts without limits before
  all), accounts of the same priority take turns, and within a turn the oldest run goes
  first. A plan's `concurrent_trainings` limits what trains at the same time (the next run
  keeps its place), the new `queued_trainings` / `BRICKDNN_QUEUE_MAX_WAITING` what may wait
  (`409` beyond it). Waiting runs reserve their budget, and the budgets are checked again when
  a run starts. Claiming is safe with several workers (`SELECT … FOR UPDATE SKIP LOCKED` and
  an advisory lock per account on PostgreSQL, a file lock on SQLite). Stop requests go
  through the `STOP` file and the database, and the worker kills what does not react, also
  for budget stops and deleted runs. Workers send a heartbeat: the runs of a worker that
  restarted (a deploy) or died go back into the queue and start again (once after a dead
  worker, at most `BRICKDNN_WORKER_MAX_RESTARTS` times in all, then they fail with a message),
  and their usage is counted once. See [docs/training.md](docs/training.md#the-training-queue)
  and [docs/scaling.md](docs/scaling.md) (what a later move to Kubernetes would change).
- **The wait is visible.** Queued runs show "Queued — position 3, starts in about 4 minutes"
  in the training panel, the run list and on the toolbar button, can be cancelled, and say in
  one line what the plan means ("Paid plans start first") where plans differ in priority; the
  panel also says before a start that the run will wait, and when no training worker is
  running at all. REST: `queue` on runs (`position`, `waiting`, `estimated_wait_seconds`,
  `estimated_start_at`, `worker_online`, `blocked_by`, `hint`), on `GET /api/training/`
  (`online`, `slots`, `running`, `waiting`, `message`) and on the training check. MCP:
  `get_training_options`, `check_training`, `start_training`, `list_training_runs` and
  `get_training_run` report the same in a few fields; `stop_training` cancels a queued run.
- **Queue overview for the operator.** *Training workers* in the Django admin and
  `manage.py training_queue` show the waiting and running trainings (account, plan, priority,
  waiting time), the workers (heartbeat, slots, memory limit, current runs) and, for the last
  seven days, the median and 95th percentile of the wait for a slot and the utilisation of
  the worker.
- **Data preparation over MCP.** AI assistants can go from a file to a trained network
  without the web app: `import_data` (CSV/JSON as text, Excel, Parquet … as base64, or a public
  link with the same checks against private addresses), `list_data_sources`,
  `get_data_source`, `create_data_prep`, `update_data_prep` (replace the steps or add, edit,
  move and remove single steps; target, task and split), `list_prep_steps` (the step catalogue,
  one line per parameter), `preview_data_prep` (step reports, column profiles, sample rows and
  the readiness check, also of unsaved changes), `apply_readiness_fix` (the suggested step by
  its `fix_id`, or `all`), `export_training_dataset` (returns the dataset reference for
  training and the feature columns `predict` takes), `get_data_prep_script` (pandas script,
  recipe or prepared rows), `list_data_preps`, `get_data_prep`, `delete_data_prep` and
  `delete_data_source`. Same service functions, roles of shared projects, limits, rate limits
  and messages as the Data tab; the server instructions describe the flow.
- **Upload links for assistants with a shell.** `create_upload_link` returns a one-time link,
  valid for 15 minutes, that takes a file with `curl -T` (or `-F file=@…`) without logging in
  (`PUT`/`POST /api/uploads/?sig=…`) and imports it as a data source or stores it as a
  training dataset, with the limits of the web app instead of the small MCP text limit. nginx
  lets large bodies through on `/api/uploads/`.
- **Plans, budgets and cost limits per account.** Trainings and predictions are measured in
  CPU time: the runner reports its own (`usage` in `state.json`, also counted when another
  server process reaped the run), predictions report theirs, and both go into a ledger
  (`UsageRecord`) per account, calendar month (UTC) and the API token or connected app that
  caused them. An account's limits come from its plan (`BRICKDNN_PLANS`, JSON:
  compute per month, storage, compute per training, trainings at a time, a priority for the
  training queue, optional prices and an overage price; `BRICKDNN_DEFAULT_PLAN`); without plans a
  server has no limits unless `BRICKDNN_BUDGET_*` are set. brickdnn.com has the plans Free
  (0.5 CPU hours, 1 GB, 15 CPU min per training), Starter, Plus and Pro. Operators change
  plans and mark accounts as unlimited in the Django admin (filters, bulk actions) or with
  `manage.py set_plan <user> --plan plus | --unlimited | --limited`; staff and superusers are
  unlimited by default (`BRICKDNN_STAFF_UNLIMITED`); unlimited accounts are still measured.
  Users can set a monthly budget for their account and for every API token and connected app
  (the budgets of agents; website login only, so an agent cannot raise its own), and with a
  paid plan switch on compute beyond the allowance with an own cap (marked in the ledger for
  billing later). Trainings (web, REST, MCP) are refused when a budget is used up, the storage
  is full or the plan allows no further training at a time; a run gets what is left as a CPU
  limit its runner enforces itself (stopping like a user stop, keeping the best weights), and
  every look at an active run stops it once a budget is used up. Active runs reserve their
  limits, so concurrent starts cannot spend the same rest. `POST /api/training/check/` and the
  MCP tool `check_training` return a rough estimate of the planned run's CPU time and whether
  the budgets allow it (a run clearly beyond its limit is refused); predictions need compute
  left, uploads and imports room in the storage quota. New: `GET`/`PATCH
  /api/account/usage/`, `PATCH` on API tokens and connections, the MCP tool `get_usage`,
  optional prices (`BRICKDNN_PRICE_PER_CPU_HOUR`, `_PER_GB_MONTH`). In the web app: *Usage &
  budget* on the account page (plan, compute and storage bars, reset date, own budget,
  overage), usage and budgets of tokens and apps on the API tokens page, the estimated
  compute and what is left in the training panel, and why a run was stopped. Usage records
  are part of the data export and are deleted with the account.
- **Agent activity: a log of everything AI agents do.** Every MCP tool call and every API
  request made with an API token or a connected app is recorded for its user: the token or
  app, the client, the tool or route, whether it worked (with the error the agent got), how
  long it took, the networks, trainings and datasets it touched and a sentence such as
  *Started training “GP 1” on dataset “houses.csv” (#3)*. Arguments are summarised and
  redacted (sizes and counts instead of CSV text and prediction inputs; never tokens or
  passwords). The new page *Agent activity* (account menu, and next to each token and app on
  the API tokens page) shows it live: new entries slide in while the page is open, repeated
  checks of a training count up in one row, failed actions are marked, details show the
  arguments, and names link to the network, training or dataset; filters for the token or app
  and for errors. Website use is not logged; each user sees only their own agents. Entries are
  kept for `BRICKDNN_AGENT_LOG_DAYS` (90) days and at most 20,000 per account, deleted by
  `purge_old_data`, exported with the account and deleted with it; revoking a token keeps its
  entries until then. API: `GET /api/agent-activity/` (filters, pages, `?after=` for polling)
  and `/api/agent-activity/summary/`. Logging happens after the response and never breaks a
  call.
- **Paid plans with Paddle** (off unless configured; [docs/billing.md](docs/billing.md)). A
  server can sell its plans as monthly subscriptions through Paddle Billing, which is the
  seller (merchant of record) and handles payment, invoice and tax. With
  `BRICKDNN_PADDLE_ENV`, `BRICKDNN_PADDLE_CLIENT_TOKEN`, `BRICKDNN_PADDLE_PRICES` and the
  secrets `BRICKDNN_PADDLE_API_KEY` and `BRICKDNN_PADDLE_WEBHOOK_SECRET` (only from the
  server's `.env`), the header links to a pricing page (`/pricing`: the plans, what they
  include, prices incl. VAT, the overage price), and the account page gets "Plan & billing":
  the plan and when it renews or ends, changing the plan (with the amount Paddle charges or
  credits), cancelling to the end of the paid month, taking that back, and Paddle's customer
  portal for payment method and invoices. Paddle's checkout opens in an overlay; Paddle.js is
  only loaded then. A paid subscription sets the account's plan: Paddle's signed
  notifications (`/api/billing/paddle/webhook/`) are processed once and in order, a
  subscription only counts for the account whose Paddle customer it belongs to, a failed
  payment keeps the plan for `BRICKDNN_PADDLE_GRACE_DAYS` (7), and a cancelled or paused
  subscription means the free plan again, without deleting anything. Unlimited accounts need
  no plan and are never changed. Compute beyond the allowance (overage) is billed to the
  subscription as a one-time charge after each month, up to the account's cap, recorded
  before Paddle is asked so that nothing is billed twice. `manage.py billing_sync` (daily in
  the cleanup service) re-reads all subscriptions, ends grace periods and bills overage;
  `--check` verifies the settings and the prices at Paddle. Deleting an account cancels its
  subscription first, the data export holds `billing.json`, and the Django admin lists
  customers, notifications and overage charges read-only. New endpoints: `GET /api/billing/`
  and `/api/billing/{subscription,checkout,change-plan,cancel,reactivate,portal,refresh}/`.
  The German templates for the privacy policy and the terms of use have sections on Paddle
  and paid plans.
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
  fitted once instead of epoch by epoch, on at most `BRICKDNN_GP_MAX_ROWS` (5,000) training
  rows for Gaussian process regression, `BRICKDNN_GPC_MAX_ROWS` (2,000) for Gaussian process
  classification and `BRICKDNN_SVM_MAX_ROWS` (50,000) for SVMs; runs report the fitted
  kernel and log marginal likelihood or the support vectors, save the fitted pipeline as a
  `.joblib` file, and predict class probabilities or values with the Gaussian process's
  uncertainty (REST, MCP and `brickdnn train`). Examples: Gaussian process regression, SVM
  classifier, periodic Gaussian process. The catalogue (`GET /api/layers/`) tells each brick's
  `family`, `output_kind` and `input_slots`; the analysis adds `output_kind`, `kernel` and
  `task` per node and `family` and `classical` for the network; `GET /api/training/` adds
  `classical`, runs `family`, and the run metrics `val_r2` and `test_r2`.
  In the editor, the palette groups the bricks under *Neural network · PyTorch* and
  *Classical · scikit-learn*, with own colours for kernels and classical models. Kernel
  connections are dashed and end at the kernel input of a Gaussian process (one labelled input
  per slot); a kernel into a data input, or data into a kernel input, is refused while
  connecting, and clicking a kernel in the palette plugs it into the selected Gaussian process
  or kernel sum. Nodes show their kernel (`RBF(ℓ=1.0)`, `Constant × RBF + White`, *default
  kernel*) or SVM kernel and `C`; the inspector shows the kernel as scikit-learn code and
  labels the options (*Optimise*, *Custom bounds*, *Fixed*), with the bounds only for custom
  bounds. The toolbar counts the hyperparameters of a Gaussian process instead of parameters,
  and the code view and export say scikit-learn. The training panel trains classical models
  without PyTorch, shows only the splits, *Max. samples* (with the row limit) and the seed,
  and shows a fitted run with its fit time, rows, fitted and starting kernel, log marginal
  likelihood or support vectors and the scores (accuracy, or RMSE and R²) instead of loss
  curves; the model downloads as `<run>.joblib` with a joblib snippet.
- **R² for regressions of neural networks**, like for classical models: the coefficient of
  determination on the validation data (per epoch) and the test data, measured against the
  mean of those rows themselves (the same number as scikit-learn's `r2_score`; with several
  outputs their mean; null when all targets of a split are equal). It is reported wherever the
  RMSE is: `val_r2` in the history, the trials and the best trial of a search, `r2` in the
  test result, the run metrics `val_r2` and `test_r2` (REST and MCP), the final message and
  `brickdnn train` (epochs, trials, the table of trials, the test line). The run view shows
  the R² next to the validation and test RMSE, a *Validation R²* curve and R² columns in the
  tables of the epochs and the trials; the list of runs shows the test R² of neural
  regressions too.
- After the first training of a user that completes, a notification asks once (per browser)
  for a coffee: Brick DNN is free, *Buy me a coffee* links to the support page.
- **Privacy basics for public servers** ([docs/privacy.md](docs/privacy.md)).
  - *Legal pages:* an imprint, privacy policy and terms of use are Markdown files the
    operator puts into `BRICKDNN_LEGAL_DIR` (on the server `/srv/brick-dnn/legal`, or in the
    repository as `deploy/sites/<domain>/legal/`, deployed with the code; settings of the
    domain such as the deletion periods in `deploy/sites/<domain>/settings.env`), shown at
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
- **More import formats in the Data tab.** Besides CSV, the Data tab imports Excel workbooks
  (`.xlsx`, `.xls`; the dialog asks which sheet when there are several), OpenDocument
  spreadsheets (`.ods`), JSON (records, objects of columns, records inside `{"data": [...]}`;
  nested objects become `a.b` columns), JSON Lines, Parquet and Feather / Arrow files, also
  packed as `.gz`, `.bz2`, `.xz` or a `.zip` file with one data file. Every format ends in the
  same typed table as a CSV file, and the downloadable pandas script reads it the same way.
  Size, row, column and cell limits apply to every format, also after unpacking (ZIP and gzip
  bombs are refused). New dependencies: openpyxl, xlrd, pyarrow, defusedxml (pyarrow adds
  about 170 MB to the backend image).
- **Import from a link.** The Data tab and `POST /api/data/sources/` with `{"url": ...}`
  import a file from a public link; share links of Google Sheets (the tab as CSV), Google
  Drive, GitHub, Hugging Face and Dropbox are turned into download links. The server refuses
  links to private, loopback, link-local and other non-public addresses (also after
  redirects; the connection goes to the checked address), limits redirects, time and size,
  and sends no credentials. `BRICKDNN_DATA_URL_IMPORT=false` switches link imports off,
  `BRICKDNN_DATA_URL_ALLOW_PRIVATE=true` allows private addresses for local installations.
  Database connections are not supported yet.

### Changed
- The backend requires **scikit-learn** (for classical models; it brings SciPy and joblib),
  which makes the Docker image about 190 MB larger. `brickdnn[train]` includes it, and the new
  extra `brickdnn[classic]` installs only it.
- Generated scripts start with a comment line that links to Brick DNN
  (`# Built with Brick DNN: https://brickdnn.com, https://github.com/alexanderKuehling/brick-dnn`):
  the PyTorch code (also from the MCP tool and with the trained weights) and the pandas script
  of the Data tab.
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
- The cap for compute beyond the allowance cannot be set below what an account has already
  used and not been billed for: lowering it afterwards would have shrunk the bill.
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
