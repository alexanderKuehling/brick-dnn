# MCP server, OAuth and API tokens

Brick DNN has a built-in [MCP](https://modelcontextprotocol.io) server, so AI assistants such
as Claude can build networks (and classical models: Gaussian processes, SVMs), check them,
generate PyTorch or scikit-learn code, save them in your account, prepare data files for
training (the Data tab), train them on your server, predict with the trained networks and hand
you their weights. The
server is at `https://<your domain>/api/mcp/`
(Streamable HTTP). Assistants sign in with **OAuth**, like "Sign in with …" on other sites;
scripts and clients without OAuth use personal **API tokens**.

## Connect with OAuth (Claude app, Claude Code and other MCP clients)

Claude (web, desktop, mobile): *Settings → Connectors → Add custom connector*, and paste
`https://<your domain>/api/mcp/`. Claude Code:

```bash
claude mcp add --transport http brick-dnn https://example.com/api/mcp/
```

then `/mcp` in Claude Code to sign in. The client opens Brick DNN in the browser: log in if
needed, check which app asks and where you will be sent back, choose whether it may also
change things (save and delete networks, upload datasets, start, stop and delete trainings)
or only look, and allow it.

Connected apps are listed on the **API tokens** page (account menu), where you can
disconnect them at any time; their access ends immediately.

How it works (OAuth 2.1 as in the MCP authorization spec):

- Without a valid token, `/api/mcp/` answers `401` with
  `WWW-Authenticate: Bearer resource_metadata="…/.well-known/oauth-protected-resource/api/mcp/"`
  (RFC 9728). That document names this site as authorization server; its metadata is at
  `/.well-known/oauth-authorization-server` (RFC 8414).
- Apps register themselves at `/api/oauth/register/` (dynamic client registration,
  RFC 7591). Redirect URIs must use HTTPS, or HTTP on `localhost`.
- `/oauth/authorize` is a page of the web app: the consent screen. Authorization codes need
  PKCE (S256) and are valid for 5 minutes, once.
- `/api/oauth/token/` issues access tokens (1 hour, only valid for the MCP server) and refresh
  tokens (rotated on every use, valid for 60 days after the last refresh). Apps can revoke
  their tokens at `/api/oauth/revoke/`.
- Scopes: `read` and `write` (write includes read). Without a scope, an app asks for both;
  the user can still allow read-only access.
- The issuer and all URLs are taken from the request (Host header and `X-Forwarded-Proto`
  behind a proxy). Set `BRICKDNN_PUBLIC_URL` (e.g. `https://brickdnn.com`) if that does not
  give the public address. With the bundled nginx, `/.well-known/oauth-*` is passed to Django.

## Connect with an API token

Log in, open the account menu → **API tokens** and create a token:

- **Full access** can also save and delete networks, upload and delete datasets, import and
  prepare data and start, stop or delete trainings. **Read only** can look at everything
  (layers, examples, your networks, data and training runs), analyse networks, generate code,
  predict with trained networks and download their weights, but change nothing.
- Choose an expiry (7, 30, 90 days, 1 year or never). You can revoke a token at any time;
  apps using it stop working immediately.
- The token (`bdnn_…`) is shown **once**. Treat it like a password.

Only a login on the website can create, list or revoke tokens, not a token itself. Send the
token as `Authorization: Bearer <token>`.

Claude Code:

```bash
claude mcp add --transport http brick-dnn https://example.com/api/mcp/ \
  --header "Authorization: Bearer bdnn_..."
```

Other clients that support remote MCP servers with headers:

```json
{
  "mcpServers": {
    "brick-dnn": {
      "type": "http",
      "url": "https://example.com/api/mcp/",
      "headers": { "Authorization": "Bearer bdnn_..." }
    }
  }
}
```

Locally the URL is `http://localhost:8000/api/mcp/` (`make backend`) or
`http://localhost:8080/api/mcp/` (Docker Compose).

## Tools

| Tool                    | Access | Destructive | What it does                                                    |
| ----------------------- | ------ | ----------- | --------------------------------------------------------------- |
| `list_layer_types`      | read   |             | All layer types with one line per parameter (`types`: full specs), kernel bricks and classical models |
| `list_examples`         | read   |             | Example networks: names and descriptions (`id`: one with its graph) |
| `analyze_network`       | read   |             | Validate a graph, infer shapes and parameter counts, explain errors |
| `generate_pytorch_code` | read   |             | PyTorch script for a valid graph (scikit-learn for a classical model) |
| `list_projects`         | read   |             | Your saved networks and the ones shared with you                |
| `get_project`           | read   |             | One saved network with its graph                                |
| `save_project`          | full   | yes         | Create a network or update one (owner, editors)                 |
| `delete_project`        | full   | yes         | Delete a network (owner only)                                   |
| `get_training_options`  | read   |             | Whether training is available, datasets (summaries), default settings, search, upload limit |
| `get_dataset`           | read   |             | One dataset: rows, task, classes and its columns, a page at a time |
| `upload_dataset`        | full   |             | Store a small CSV file (sent as text) as a training dataset     |
| `create_upload_link`    | full   |             | One-time link to upload a local file with curl (data source or training dataset) |
| `delete_dataset`        | full   | yes         | Delete one of your own datasets                                 |
| `list_data_sources`     | read   |             | Your imported data files, and what `import_data` accepts        |
| `get_data_source`       | read   |             | One imported file: columns with types, import report, first rows |
| `import_data`           | full   |             | Import a file (CSV/JSON text, base64 for Excel/Parquet …, or a public link) |
| `delete_data_source`    | full   | yes         | Delete an imported file (owner; not while a preparation uses it) |
| `list_data_preps`       | read   |             | Your data preparations and the ones shared with you             |
| `get_data_prep`         | read   |             | One preparation: source, settings, steps, exported datasets     |
| `list_prep_steps`       | read   |             | The step catalogue with one line per parameter (`type`: help texts) |
| `preview_data_prep`     | read   |             | Run a preparation (also unsaved changes): step reports, profiles, sample rows, readiness with fixes |
| `get_data_prep_script`  | read   |             | The pandas script (or the recipe, or prepared rows as CSV)      |
| `create_data_prep`      | full   |             | Start a preparation: target, task, split, first steps            |
| `update_data_prep`      | full   | yes         | Change steps (replace, add, edit, move, remove) and settings (owner, editors) |
| `apply_readiness_fix`   | full   |             | Add the step that fixes a readiness issue (`fix_id`, or `all`)  |
| `export_training_dataset` | full |             | Turn a ready preparation into a training dataset (owner, editors) |
| `delete_data_prep`      | full   | yes         | Delete a preparation (owner only)                               |
| `check_training`        | read   |             | Does a network fit a dataset (optionally with the settings)? With `fixes` for the Input and last layer, the estimated CPU time and whether the budgets allow it |
| `get_usage`             | read   |             | This month's usage, the plan and its limits, what is left (also of your token's or app's budget), when it resets |
| `start_training`        | full   |             | Train in the background                                         |
| `list_training_runs`    | read   |             | Your training runs, one line each with status and metrics, for regressions RMSE and R² (`limit`, `status`, `project_id`) |
| `get_training_run`      | read   |             | One run as a summary (`include`: history, graph, plan, log, trials, preprocessing) |
| `stop_training`         | full   |             | Stop a run after the current batch (starter, project editors)   |
| `delete_training_run`   | full   | yes         | Delete a run with its weights (starter, project owner)          |
| `get_weights_download`  | read   |             | Short-lived download link for the trained weights, with model code and preprocessing |
| `predict`               | read   |             | Run a trained network on up to 64 samples                       |

Read-only tokens and apps allowed read-only access only see the read tools. Networks use the [graph format](graph-format.md).

**Compact answers.** Tool results go into the assistant's context, so they are summaries by
default, and every capped list says how long it is and how to see more:

- `get_training_options` lists each dataset with what it takes to choose one (rows, features,
  target, task, number of classes, size, `reference`), no column lists; `get_dataset {id,
  columns_limit?, columns_offset?}` pages through the columns (50 at a time) and shows the
  class names (at most 50).
- `get_training_run` is a summary: status, settings, epochs done, the best epoch, final and
  test metrics, rows per split, tuning (trials, best trial), the classical fit, and the end of
  the log of a failed run. `include: ["history", "graph", "plan", "log", "trials",
  "preprocessing"]` adds those parts; `history` has at most 50 evenly spaced epochs plus the
  best and the last, `log` the last `log_lines` lines (default 20, at most 200).
  `list_training_runs {limit?, status?, project_id?}` shows the newest 20 runs, one line each,
  with the total.
- `list_layer_types` gives one line per parameter (`types` gives the full specs with help
  texts), `list_examples` names and descriptions (`id` gives one example with its graph),
  `check_training` at most 50 class names.
- Data tools show wide tables in pages: 50 columns (`columns_offset` for more), profiles and
  sample rows of 20 of them (`columns` chooses which), and readiness issues name at most 10
  columns ("… and 812 more").

A dataset with 800 columns took 17 KB in `get_training_options` before and 3.4 KB now (most of
it the training settings); a run with 200 epochs took 71 KB in `get_training_run` and now
takes 1 KB. The REST API and the web app keep their full answers.

**Classical models.** The same tools work for Gaussian processes and support vector machines
([graph-format.md](graph-format.md#classical-models)): the instructions and
`list_layer_types` explain the kernel bricks (`RBF`, `WhiteKernel`, … combined with
`KernelSum` and `KernelProduct`) and the models; `analyze_network` reports the kernel
expression; `generate_pytorch_code` returns a scikit-learn script; `get_training_options`
says under `classical` whether they can be trained and on how many rows at most;
`start_training` fits them once (only `val_split`, `test_split`, `max_samples` and `seed`
apply, no `tuning`); `get_training_run` shows the fitted kernel or the support vectors in
`state.classical` (see [training.md](training.md#classical-models)).
Everything happens as the token's owner: an assistant sees and changes only that user's
networks, datasets and runs and the ones shared with them, with the same roles, limits and
validation as the web app (see "Sharing" in [api.md](api.md#saved-networks-login-required)).
Destructive tools are marked as such for the client (`destructiveHint`), so it can ask you
before it uses them.

### Budgets

Trainings and predictions use compute (CPU time), which the account's plan and the budgets of
the account and of each token and app limit per month (see
[training.md](training.md#usage-and-budgets)). `get_usage` shows what is left, under `caller`
also the budget of the token or app the assistant uses; `check_training` (with `config` and
`tuning`) returns `estimate`, a rough estimate of the CPU time, and `budget`: whether a start
would be allowed (`allowed`, `reason`), a `warning` if the run will probably be stopped by its
limit, and `run_limit_cpu_minutes`, the most it may use. A refused start or prediction is a
tool error that names the budget, how much is used, when it resets and who can raise it; a
training that runs out stops by itself and keeps its weights (`get_training_run`:
`usage.stop_reason` `budget` or `run_limit`). Only the user can change budgets, in the web app
(account page, API tokens page): a token cannot raise its own. Checks, analyses and listings
never count.

### Deleting

- `delete_project`: only the owner deletes a network; editors and viewers get an error that
  tells them to leave it in the web app instead (*My networks* → ⋯ → *Leave*). People
  editing it live are disconnected, as when it is deleted in the web app. Training runs
  started in it are kept for who started them.
- `delete_training_run`: who started the run and the owner of its project. An active run is
  killed first.
- `delete_dataset`: your own datasets only (not the ones others shared with you), and not
  while a training uses them.

### Uploading a dataset

Which way to send a file:

| File | Tool |
| ---- | ---- |
| Small (up to `BRICKDNN_MCP_UPLOAD_MB`, 5 MB) | as text: `upload_dataset` (a CSV file ready for training) or `import_data` (any file to prepare first; binary files as base64) |
| Larger, on a computer where the assistant can run a shell command | `create_upload_link`, then `curl` |
| Public on the internet | `import_data {url}` |
| Anything else | ask the user to upload it in the web app |

`upload_dataset {name, csv, project_id?}` takes the content of a CSV file as text: features in
all columns but the last, the target in the last one (see [training.md](training.md)). It is
stored and checked exactly like an upload in the web app and returned like
`get_training_options` lists datasets, plus `reference` (`{"kind": "upload", "id": …}`) to
train on it. With `project_id` (a project you own or edit) it is shared there right away.
CSV text can have up to `BRICKDNN_MCP_UPLOAD_MB` (default 5 MB, at most
`BRICKDNN_MAX_UPLOAD_MB`); the MCP endpoint accepts requests of about twice that size, while
other JSON requests stay limited to 1 MB. The bundled nginx lets requests to `/api/mcp` through
up to 12 MB.

**Upload links.** Assistants cannot put megabytes into a tool call, but those with a shell
(Claude Code and others) can run curl. `create_upload_link {purpose, filename, name?,
project_id?, …import options}` returns a one-time link, valid for 15 minutes, and the commands
to use it:

```bash
curl --fail-with-body -T "sales.csv" "https://example.com/api/uploads/?sig=…"   # the file as the body
curl --fail-with-body -F "file=@sales.csv" "https://example.com/api/uploads/?sig=…"  # or multipart
```

`purpose` `data_source` imports the file into the Data tab like `import_data` (every format it
reads, with its options: `delimiter`, `decimal`, `header`, `encoding`, `missing_values`,
`sheet`); `training_dataset` stores a CSV or ZIP file like `upload_dataset` (optionally shared
in `project_id`). The answer of curl is the JSON the inline tool returns: the new source with
its columns, or the dataset with its `reference`. The upload goes through the same functions,
limits (`BRICKDNN_DATA_MAX_UPLOAD_MB` and `BRICKDNN_MAX_UPLOAD_MB`, not the MCP text limit),
checks and rate limit (`BRICKDNN_UPLOAD_RATE`) as in the web app. The file is read under the
`filename` of the link, whose ending decides the format.

The link works **without logging in**. Its signature binds it to the user who asked for it,
the purpose, the file name, the options and a random id; it works once (a failed upload, e.g.
an unreadable file, can be sent again), the user must still be active, and sharing in a
project needs editing rights in it at upload time. Only tokens and apps with write access get
the tool. The signature is in the query string, which neither nginx nor gunicorn writes to
its access log. Treat the link like a password while it is valid.

## Preparing data

Real tables need work before a network can learn from them: missing values, categories as
text, dates, outliers, several files. The Data tab ([data-preparation.md](data-preparation.md))
does that step by step, and the same is available over MCP, with the same roles (viewers of a
shared preparation look, editors change and export, only the owner deletes), limits, rate
limits and validation as the web app; changes show up in the web app right away.

1. **Import:** `import_data` takes the text of a CSV, TSV, JSON or JSON Lines file
   (`content`), a binary file as base64 (`content_base64` with `filename`: Excel, OpenDocument,
   Parquet, Feather, packed files), or a public link (`url`, downloaded with the checks of
   [link imports](data-preparation.md#security-of-link-imports); off with
   `BRICKDNN_DATA_URL_IMPORT=false`). Separator, decimal mark, header, encoding and
   missing-value markers are detected. Larger local files go through `create_upload_link`.
2. **Prepare:** `create_data_prep {source_id, target, task?, split?, steps?}` answers with the
   report of every step and the **readiness**: errors that block the export, warnings (target
   leakage, duplicates, rare classes …) and suggestions, each with a `fix_id` where a step
   fixes it. `preview_data_prep` adds column profiles, the target and sample rows, also for
   unsaved `steps`/`settings`.
3. **Fix:** `apply_readiness_fix {id, fix_id}` adds the suggested step (`"all"`: the fixes of
   all errors, like *Fix all*). `update_data_prep` changes the steps: the whole list
   (`steps`), or `add_steps` (`{type, params?, after?}`), `edit_steps` (`{id, params?,
   enabled?}`, params are merged, `null` resets one), `move_steps` (`{id, after}`) and
   `remove_steps`; and the settings (`target`, `task`, `split`, or `settings`). New steps get
   the ids `s1`, `s2` … and the defaults of their type, as in the web app. A step with wrong
   settings fails in the report with the reason and the columns it saw; `list_prep_steps`
   explains every step type.
4. **Export:** `export_training_dataset` turns the ready preparation into a training dataset
   and returns its `reference`, the task, the target and the feature columns in order.
5. **Train and predict:** `check_training` and `start_training` with the `reference`; `predict`
   takes the prepared feature columns (e.g. rows from the sample of `preview_data_prep`).
   `get_data_prep_script` returns the pandas script whose `prepare(df)` prepares new raw data
   the same way.

```text
import_data {content: "<sales.csv>", filename: "sales.csv"}         → source 7: date (datetime), store (categorical), temperature (numeric, 6 missing), sales …
create_data_prep {source_id: 7, target: "sales"}                    → prep 3, readiness: 3 errors with fix_ids (one-hot store, date parts, impute temperature)
apply_readiness_fix {id: 3, fix_id: "all"}                          → steps s1–s3 added, ready: regression, 8 features, split 84/18/18
preview_data_prep {id: 3}                                           → profiles, sample rows with their split
export_training_dataset {id: 3}                                     → reference {"kind": "upload", "id": 12}, features date_year … temperature
check_training {graph: Input [8] → Linear 16 → ReLU → Linear 1, dataset: {"kind": "upload", "id": 12}} → ok
start_training {…, config: {epochs: 30}} · get_training_run {id}     → completed, val_rmse …
predict {run_id, inputs: [{"date_year": 2024, "date_month": 3, …}]} → values in the units of sales
```

### Downloading the weights

Binary files cannot go through MCP, so `get_weights_download {id}` returns a link,
`/api/training/runs/<id>/weights/?sig=…`, that downloads the weights (`.pt`, a PyTorch
`state_dict`) **without logging in** for 15 minutes. The signature binds it to the run and to
the user who asked for it; on download, that user must still be active and allowed to see the
run (someone removed from a project can no longer use their links). Treat the link like a
password while it is valid. Along with it come `model_code` (the `NeuralNetwork` class the
weights belong to), `preprocessing` as recorded by the training (feature order and
standardisation, or image size, channels and normalisation, and the class names), `inputs`
(the same in words) and `how_to_use`, a Python snippet that loads the weights and predicts.
For a classical model the file is the fitted scikit-learn pipeline (`.joblib`, see
[training.md](training.md#classical-models)), `model_code` is its `make_model()` and the
snippet loads the file with joblib.

### Predictions

`predict {run_id, inputs, input_scale?, top_k?}` runs the trained network of a finished or
stopped run on the server and answers one prediction per sample: for classification the
`class`, its `index` and `probability` and the `top` classes (`top_k`, default 3, at most
10); for regression the `value` in the units of the target column. Classical models take
the same rows; a Gaussian process regressor adds `std`, the uncertainty of each value.

| Trained on | `inputs` (at most 64 samples)                                       | `input_scale` |
| ---------- | ------------------------------------------------------------------- | ------------- |
| CSV file   | One row per sample: `{"column": value, …}` or a list of the feature values in column order, as in the CSV file (standardised like in training) | `raw` (default) or `normalized` (already standardised) |
| Images     | One image per sample as a nested array of channels × height × width, e.g. `1×28×28` (grayscale also `28×28`), or in the shape of the Input layer (e.g. `784` values) | required: `raw` (pixels 0–255), `unit` (0–1) or `normalized` (already normalised like in training) |
| Demo data  | One nested array per sample in the shape of the Input layer, used as it is | – |

For CSV datasets made in the Data tab, the network takes the prepared columns (after the steps
of the preparation), not the raw data. Inputs are checked first and errors say what the
network expects; at most 250,000 numbers per request. The network then runs in a separate
process, like a training (CPU, lower priority, 30 seconds at most, one prediction at a time on
the server by default: `BRICKDNN_MAX_PREDICTIONS`). Everyone who can see the run may predict,
also with read-only access. The same is available as
[`POST /api/training/runs/<id>/predict/`](api.md#training-login-required).

## Activity log

Everything an assistant or script does with your tokens and connected apps is listed on the
page **Agent activity** (account menu; on the API tokens page next to each token and app), as
it happens: the page asks for new entries every few seconds while it is open.

- **What is logged:** every MCP tool call, and every other API request made with an API token
  (or an app's token), also when it is refused (read-only token, app token outside the MCP
  server). Using the website is not logged.
- **Each entry:** the token (name and start, `bdnn_Ab3dEf9h…`) or app, the client's user agent,
  the tool or REST route, whether it worked (with the error message the assistant got), how
  long it took, the networks, training runs, datasets, data files and preparations it touched
  (with id and name, linked) and a sentence such as *Started training “GP 1” on dataset
  “houses.csv” (#3)*.
- **Arguments are summarised and redacted:** CSV text becomes its size, lines and columns
  (`48 KB, 801 lines, 9 columns`), prediction inputs their number (`3 samples with 8
  values`), graphs their layers (`5 layers (Input, Linear ×2, …), 4 connections`), long texts
  and descriptions their length, links lose their query. Keys such as `password`, `token`,
  `secret`, `sig` and anything that looks like a token (`bdnn_…`, a JWT) are stored as
  `[redacted]`. Names are capped at 120 characters.
- **Who sees it:** each user sees what their own tokens and apps did, nothing else; the
  owner of a shared network does not see other members' agents (their changes are in the
  network's history as usual). Only a website login can read the log, not an API token.
- **How long:** `BRICKDNN_AGENT_LOG_DAYS` days (default 90), at most 20,000 entries per account;
  `purge_old_data` deletes older ones daily. `0` turns the log off. Entries outlive a revoked
  token or a disconnected app (with its name) until they expire; they are part of the data
  export and deleted with the account ([privacy.md](privacy.md)).
- Entries are written after the answer has been sent: logging never delays or breaks a tool
  call. The API is [`GET /api/agent-activity/`](api.md#agent-activity).

## Details

- JSON-RPC 2.0 over HTTP `POST`, answered with `application/json`. There are no sessions and
  no event streams (`GET` answers `405`); notifications get `202`. Protocol versions
  `2025-11-25`, `2025-06-18`, `2025-03-26` and `2024-11-05`.
- Requests from browsers must come from this site or a `CORS_ALLOWED_ORIGINS` origin.
- Rate limits: 600 MCP requests per hour per user (`BRICKDNN_MCP_RATE`), plus the usual limits
  for starting trainings (`BRICKDNN_TRAINING_RATE`), uploads (`BRICKDNN_UPLOAD_RATE`) and
  predictions (`BRICKDNN_PREDICT_RATE`, 120 per hour; only calls that reach the network count);
  app registration, token and revocation requests: 120 per minute per IP address
  (`BRICKDNN_OAUTH_RATE`).
- Only SHA-256 hashes of tokens, refresh tokens, codes and client secrets are stored. Admins
  can see and revoke tokens, apps and connections in the Django admin but never see a token.
- The web app may not be shown in frames of other sites (`frame-ancestors 'none'`), so the
  consent screen cannot be overlaid.
