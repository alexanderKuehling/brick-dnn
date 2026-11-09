# REST API

All endpoints live under `/api/`. Requests and responses are JSON. Code generation does not
need an account; saved networks, training and data preparation do.

## Code generation (public)

| Method | Path              | Description                                                   |
| ------ | ----------------- | ------------------------------------------------------------- |
| GET    | `/api/`           | Index of endpoints and version                                |
| GET    | `/api/health/`    | `{"status": "ok", "version": "1.1.0"}`                        |
| GET    | `/api/layers/`    | `{"categories": [...], "layers": [LayerSpec, ...]}`           |
| GET    | `/api/examples/`  | Example networks `[{id, name, description, graph}]`           |
| POST   | `/api/analyze/`   | Validate a graph, infer shapes                                |
| POST   | `/api/generate/`  | Generate PyTorch code                                         |

### `POST /api/analyze/`

Body: `{"graph": {...}}` (or the bare graph). Always answers `200`:

```json
{
  "valid": false,
  "errors": [{ "node": "fc1", "message": "Input features is set to 50, but the incoming tensor has 100 features in its last dimension. Clear the field to infer it automatically." }],
  "warnings": [],
  "nodes": {
    "input": { "type": "Input", "output_shape": [100], "dtype": "float", "num_params": 0, "params": { "shape": [100], "dtype": "float" } },
    "fc1": { "type": "Linear", "output_shape": null, "dtype": null, "num_params": null, "params": { "in_features": 50, "out_features": 64, "bias": true } }
  },
  "inputs": ["input"],
  "outputs": ["fc1"],
  "total_params": null
}
```

`node` is `null` for errors about the whole graph. `params` contains the resolved values,
including inferred `auto` parameters; parameters that have no effect with the other settings
(the UI hides them, e.g. `padding_size` unless `padding` is `custom`) are left out. `inputs`
and `outputs` list the Input layers and the layers without outgoing connections, also when
the graph has errors.

### `POST /api/generate/`

```json
{
  "graph": { "nodes": [...], "edges": [...] },
  "options": { "class_name": "MyNet", "include_main": true, "include_training": true, "filename": "my_net" }
}
```

- `200 {"code": "...", "filename": "my_net.py"}`
- `400 {"detail": "...", "analysis": {...}}` if the graph has errors
- `?download=1` returns the file itself (`Content-Disposition: attachment`).

```bash
curl -X POST http://localhost:8080/api/generate/?download=1 \
  -H "Content-Type: application/json" -d @my_network.json -o model.py
```

## Accounts (JWT)

| Method | Path                       | Body                               | Response                    |
| ------ | -------------------------- | ---------------------------------- | --------------------------- |
| POST   | `/api/auth/register/`      | `{username, email, password}`      | `201 {user, access, refresh}` |
| POST   | `/api/auth/token/`         | `{username, password}`             | `{access, refresh}`         |
| POST   | `/api/auth/token/refresh/` | `{refresh}`                        | `{access, refresh}` (rotated) |
| POST   | `/api/auth/logout/`        | `{refresh}`                        | blacklists the refresh token |
| GET    | `/api/auth/me/`            | –                                  | `{id, username, email, date_joined}` |

Send the access token as `Authorization: Bearer <access>`. Access tokens live 30 minutes,
refresh tokens 7 days. The e-mail address is required for new accounts and can be used by one
account only. Registration can be disabled with `BRICKDNN_ALLOW_REGISTRATION=false`.

## API tokens

For scripts and AI assistants: personal tokens that do not expire after 30 minutes. Send them
like an access token, `Authorization: Bearer bdnn_…`; requests then act as the token's owner.
Read-only tokens may only use `GET` and the `POST` endpoints that change nothing (analyze,
generate, training check, MCP). The MCP server for AI assistants is described in
[mcp.md](mcp.md).

| Method | Path                         | Body                                         | Response                       |
| ------ | ---------------------------- | -------------------------------------------- | ------------------------------ |
| GET    | `/api/auth/api-tokens/`      | –                                            | `[ApiToken]`, newest first     |
| POST   | `/api/auth/api-tokens/`      | `{name, read_only?, expires_in_days?}`       | `201 {...ApiToken, token}`     |
| DELETE | `/api/auth/api-tokens/<id>/` | –                                            | `204`, the token stops working |

`ApiToken` is `{id, name, prefix, read_only, created_at, last_used_at, expires_at}`.
`expires_in_days` is `7`, `30`, `90`, `365` or `null` (never, the default). The token itself
is only returned by `POST`; only its hash is stored. At most 20 tokens per user. These three
endpoints need a website login (JWT): an API token cannot manage tokens.

## OAuth (MCP sign-in)

For MCP clients; details in [mcp.md](mcp.md).

| Method      | Path                                        | Description                                    |
| ----------- | ------------------------------------------- | ---------------------------------------------- |
| GET         | `/.well-known/oauth-protected-resource/api/mcp/` | Protected resource metadata (RFC 9728)    |
| GET         | `/.well-known/oauth-authorization-server`   | Authorization server metadata (RFC 8414)       |
| POST        | `/api/oauth/register/`                      | Dynamic client registration (RFC 7591)         |
| GET / POST  | `/api/oauth/authorize/`                     | Check a request / the user's decision (consent page, JWT) |
| POST        | `/api/oauth/token/`                         | `authorization_code` (PKCE) and `refresh_token` grants |
| POST        | `/api/oauth/revoke/`                        | Revoke a token (RFC 7009)                      |
| GET         | `/api/auth/connections/`                    | Apps you connected (JWT)                       |
| DELETE      | `/api/auth/connections/<id>/`               | Disconnect an app, revoking its tokens (JWT)   |

## Saved networks (login required)

| Method        | Path                                   | Description                                          |
| ------------- | -------------------------------------- | ---------------------------------------------------- |
| GET           | `/api/projects/`                       | Your networks and the ones shared with you, newest first (`?scope=mine` / `shared`) |
| POST          | `/api/projects/`                       | `{name, graph, description?}`                        |
| GET           | `/api/projects/<id>/`                  | One network                                          |
| PUT / PATCH   | `/api/projects/<id>/`                  | Update (owner, editors); send `version` to avoid overwriting others |
| DELETE        | `/api/projects/<id>/`                  | Delete (owner)                                       |
| POST          | `/api/projects/<id>/duplicate/`        | Copy into your account                               |
| GET / POST    | `/api/projects/<id>/members/`          | List members / invite `{user: username or e-mail, role}` (owner) |
| PATCH / DELETE| `/api/projects/<id>/members/<member>/` | Change the role (owner) / remove (owner) or leave (the member) |
| POST          | `/api/projects/<id>/link/`             | `{enabled, regenerate?}`: view link on or off (owner) |
| GET           | `/api/shared/<token>/`                 | A network shared by link (no login needed)           |
| POST          | `/api/shared/<token>/duplicate/`       | Copy a linked network into your account              |

Graphs may be saved with errors (work in progress) but are limited to 500 layers and 2000
connections.

**Sharing.** The owner shares a network with other users as `editor` (may change it) or
`viewer` (may look), or turns on a view link (`/share/<token>` in the web app). Every project
in a response has `owner`, `role` (`owner`, `editor` or `viewer`), `version`, `updated_by`,
`shared`, `link_access` and, for the owner, `share_token`. Turning the link off invalidates
it; turning it on again creates a new one.

**Conflicts.** `version` goes up with every change. A `PATCH` with `version` that is not the
current one answers `409` with `{detail, project}` (the current state) and saves nothing;
without `version`, the save overwrites.

## Training (login required)

Training runs on the server in a separate process per run, see [training.md](training.md).
`GET /api/training/` is public so that the UI can show whether training is available.

| Method | Path                                   | Description                                         |
| ------ | -------------------------------------- | --------------------------------------------------- |
| GET    | `/api/training/`                       | Availability, devices, built-in datasets (`train_size` training and `test_size` official test images), limits |
| GET    | `/api/training/datasets/`              | Your uploaded and prepared datasets                 |
| POST   | `/api/training/datasets/`              | Upload `file` (`.csv` or `.zip`), optional `name` (multipart) |
| GET / DELETE | `/api/training/datasets/<id>/`   | One dataset / delete it with its files              |
| POST   | `/api/training/check/`                 | `{graph, dataset}`: does the network fit the data?  |
| GET    | `/api/training/runs/`                  | Your runs, newest first                             |
| POST   | `/api/training/runs/`                  | Start a run `{name, graph, dataset, config, tuning?}` |
| GET    | `/api/training/runs/<id>/`             | Run with live state (history, progress, log)        |
| POST   | `/api/training/runs/<id>/stop/`        | Stop after the current batch (the best weights so far are kept) |
| DELETE | `/api/training/runs/<id>/`             | Delete the run (stops it first)                     |
| GET    | `/api/training/runs/<id>/weights/`     | Download the trained `state_dict` (`.pt`)           |

`dataset` is one of `{"kind": "random"}` (demo data), `{"kind": "builtin", "id": "mnist"}`
(`mnist`, `fashion_mnist`, `cifar10`) or `{"kind": "upload", "id": 3}`. `config` may contain
`epochs`, `batch_size`, `learning_rate`, `optimizer` (`adam`, `adamw`, `sgd`), `weight_decay`,
`val_split`, `test_split` (0 = no test set; validation and test together at most 0.8; built-in
datasets take `val_split` of their training images and always test on their official test
images, `test_split` is not used for them),
`device` (`auto`, `cpu`, `cuda`, `mps`), `max_samples`, `seed`, `patience` (early stopping:
stop when the validation loss has not improved for this many epochs, 1–1000; `null`, the
default, trains every epoch) and `restore_best` (default `true`: test and save the weights of
the epoch with the lowest validation loss instead of the last one); missing values use the
defaults.

`tuning` (optional) turns the run into a hyperparameter search, see
[training.md](training.md#hyperparameter-search):

```json
{
  "method": "random",
  "trials": 20,
  "space": {
    "learning_rate": { "min": 0.0001, "max": 0.01 },
    "batch_size": [32, 64, 128],
    "optimizer": ["adam", "sgd"],
    "weight_decay": [0, 0.0001]
  }
}
```

`method` is `random` (default) or `grid` (every combination; lists only, at most 100).
`learning_rate` and `weight_decay` take a list or a range `{min, max}` (log-uniform, min > 0),
`batch_size` and `optimizer` a list (at most 20 values each). Settings that are not in `space`
keep their `config` value; `{}` uses the default space (learning rate 1e-4…1e-2, batch size
32/64/128). `GET /api/training/` lists the limits and defaults under `tuning`.

```bash
TOKEN=...   # access token from /api/auth/token/
curl -X POST http://localhost:8080/api/training/runs/ \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"name": "MLP", "graph": '"$(cat mlp.json)"', "dataset": {"kind": "builtin", "id": "mnist"}, "config": {"epochs": 3}}'
```

`POST /api/training/check/` and `POST /api/training/runs/` answer incompatible combinations with
explanations, e.g. `400 {"detail": "...", "errors": ["The network outputs 5 values, but MNIST has
10 classes. ..."], "warnings": []}`. Starting a run while another one is active gives `409`, and
`503` when PyTorch is not installed or training is disabled.

A run's `status` is `starting`, `preparing` (loading or downloading data), `running`,
`completed`, `stopped` or `failed`. `GET /api/training/runs/<id>/` includes `state` with the
per-epoch `history` (`train_loss`, `val_loss`, `train_accuracy`, `val_accuracy`, `val_rmse`),
the current `progress`, the `best` epoch, `weights_epoch` (the epoch whose weights are tested
and saved, set when the epochs are over), `stopped_early`, the `test` result, `preprocessing`
and the last log lines. Poll it about once per second while a run is active.

Every run also has `metrics` (the last epoch while it runs, then the kept epoch, plus
`test_loss`, `test_accuracy`, `test_rmse`), `tuning` (the normalised search settings or
`null`), `trials_done` and `best_trial` (`{trial, params, val_loss, val_accuracy, val_rmse,
weights_epoch}`). In a search, `state.trials` lists every trial with its `params`, `status`,
the metrics of its kept weights, `weights_epoch` and `stopped_early`; `history`, `best` and
`metrics` belong to the running trial and, at the end, to the best one. The file format is
documented in `backend/brickdnn/training/state.py`.

## Data preparation (login required)

The Data tab imports CSV files, combines and cleans them step by step and turns the result into
a training dataset. The work is done on the server with pandas (`pip install "brickdnn[data]"`),
see `backend/brickdnn/data/`. `GET /api/data/` and `GET /api/data/steps/` are public; all JSON
shapes are documented in `frontend/src/api/types.ts` (section "Data preparation").

| Method | Path                                   | Description                                         |
| ------ | -------------------------------------- | --------------------------------------------------- |
| GET    | `/api/data/`                           | Availability and limits (`DataStatus`)              |
| GET    | `/api/data/steps/`                     | All step types with their settings (`StepCatalog`)  |
| GET    | `/api/data/sources/`                   | Your imported CSV files, newest first               |
| POST   | `/api/data/sources/`                   | Import `file` (`.csv`, `.tsv`, `.txt`) with import options (multipart) |
| GET / PATCH / DELETE | `/api/data/sources/<id>/` | One source / rename it (`{name}`) / delete it with its files |
| GET    | `/api/data/preps/`                     | Your preparations, last changed first               |
| POST   | `/api/data/preps/`                     | `{source, name?, steps?, settings?}`                |
| GET / PATCH / DELETE | `/api/data/preps/<id>/`   | One preparation / change it / delete it             |
| POST   | `/api/data/preps/<id>/preview/`        | Run the (unsaved) pipeline: reports, profile, sample, readiness |
| POST   | `/api/data/preps/<id>/export/`         | Create a training dataset from the saved preparation `{name?}` |
| GET    | `/api/data/preps/<id>/download/?format=…` | `csv` (prepared data), `recipe` (JSON) or `python` (pandas script) |

**Import.** Besides `file`, the form may contain `name`, `delimiter` (`auto`, `,`, `;`, a tab,
`|`), `decimal` (`auto`, `.`, `,`), `header` (`auto`, `yes`, `no`), `encoding` (`auto`, `utf-8`,
`cp1252`, `latin-1`) and `missing_values` (a JSON list of cell values that mean "missing", e.g.
`["-999", "n/a"]`). Left out means "detect". The answer (`201`) is the source with its typed
`columns` (`{name, kind, missing}`), the requested `import_options` and the `report` of what was
detected (`delimiter`, `decimal`, `header`, `encoding`, `rows`, `columns`, `warnings`). Files
larger than `BRICKDNN_DATA_MAX_UPLOAD_MB` (default 200) are refused (`413`), files with more than `BRICKDNN_DATA_MAX_ROWS`
rows (or more than `BRICKDNN_DATA_MAX_COLUMNS` columns, if set) are refused (`400`). A source that a preparation
uses as its main source cannot be deleted (`409 {"detail": "Used by the preparation “Churn”;
delete it first."}`).

```bash
curl -X POST http://localhost:8080/api/data/sources/ -H "Authorization: Bearer $TOKEN" \
  -F file=@customers.csv -F delimiter=";" -F decimal="," -F 'missing_values=["-999"]'
```

**Preparations.** A preparation is a main `source`, a list of `steps`
(`{"id": "s1", "type": "impute", "enabled": true, "params": {...}}`, types from
`/api/data/steps/`; combine steps such as `join` and `append` come first) and `settings`:

```json
{ "target": "churn", "task": "auto", "split": { "val": 0.15, "test": 0.15, "seed": 42, "stratify": true } }
```

Missing settings get these defaults. The answer adds `source_name` and `datasets` (the training
datasets exported from it, `{id, name, created_at}`, newest first). Invalid steps or settings
give `400 {"detail": "...", "steps": ["..."]}`; the source must be one of your own.

**Preview.** `POST /api/data/preps/<id>/preview/` with an optional body

```json
{ "steps": [...], "settings": {...}, "upto": "s3", "sample_rows": 5000, "compare": true }
```

previews unsaved `steps`/`settings` (the saved ones if left out) without saving them. The answer
(`PreviewResult`) has one report per step (`status` `ok`, `error`, `skipped`, `disabled` or
`not_run`, a one-sentence `summary`, `warnings`, `details`, the `columns` after the step,
`added`/`removed` columns, what the step learned in `fitted`), the resolved `task`, the
`profile` of the result (statistics, histograms, correlations, target analysis and warnings
with suggested fixes), a seeded random `sample` of rows, the training `readiness` and, with
`compare`, the profile `before` the cleaning steps. `upto` shows the data after that step.
`sample_rows` is clamped to 10 … `max_sample_rows` (default `default_sample_rows`). A failing
step does not fail the request: its report says what is wrong and the later steps are
`skipped`. Join and append steps can only read your own sources.

The train/validation/test split is assigned right after the combine steps, and every step that
learns values (means, bounds, categories …) learns them from the training rows only.

**Export.** `POST /api/data/preps/<id>/export/` runs the saved preparation and stores the result
as a training dataset (`201`, a `Dataset` as in `/api/training/datasets/`). Data that is not
ready gives `400 {"detail": "...", "issues": [...]}` (the readiness issues, with fixes). The
dataset is a CSV file with the numeric features and the target last (class indices for
classification) plus `split.bin`, the split of the preparation. Its `info.predefined_split` is
`[train, validation, test]` rows and `info.class_names` the original labels; `prepared_from` is
`{id, name}` of the preparation (`null` for uploads). Training then uses this split instead of
`val_split`/`test_split`, so the test rows stay unseen.

**Downloads** are attachments named after the preparation: `<name>.csv` (the prepared table
with a last column `split`), `<name>.recipe.json` (sources, settings, steps with their fitted
values) and `<name>_prepare.py` (a standalone pandas script that prepares new data the same way).

All data endpoints except `GET /api/data/` answer `503` when data preparation is switched off
(`BRICKDNN_DATA_PREP=false`) or pandas is not installed; `GET /api/data/` then says why
(`available: false`, `reason`).

## Rate limits

Anonymous clients: 300 requests/minute, logged-in users: 600/minute, login and registration:
20/minute, starting training runs: 120/hour, dataset uploads, CSV imports and exports of
preparations: 60/hour, previews and downloads of preparations: 1200/hour, MCP requests:
600/hour, OAuth registration, token and revocation requests: 120/minute. Adjust with
`BRICKDNN_ANON_RATE`, `BRICKDNN_USER_RATE`, `BRICKDNN_AUTH_RATE`, `BRICKDNN_TRAINING_RATE`,
`BRICKDNN_UPLOAD_RATE`, `BRICKDNN_DATA_RATE`, `BRICKDNN_MCP_RATE`, `BRICKDNN_OAUTH_RATE` and
`BRICKDNN_SHARING_RATE` (invitations and link changes, 120/hour)
(e.g. `100/minute`).
