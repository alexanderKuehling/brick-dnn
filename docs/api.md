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
| POST   | `/api/generate/`  | Generate PyTorch code (scikit-learn for classical models)     |

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

Every node also has `output_kind` (`tensor` or `kernel`: what its outgoing edges carry),
`kernel` (kernel bricks and Gaussian processes: the kernel as scikit-learn code, otherwise
`null`) and `task` (classical models: `regression` or `classification`). The top level adds
`family` (`neural` or `classical`) and `classical` (`{"node", "type", "estimator", "task",
"kernel"}` for a network with a classical model, else `null`). For example a Gaussian process
without a kernel brick:

```json
"gp": { "type": "GaussianProcessRegressor", "output_shape": [1], "dtype": "float", "num_params": 3,
        "params": { "alpha": 1e-10, "normalize_y": true, "n_restarts_optimizer": 0 },
        "output_kind": "tensor", "task": "regression",
        "kernel": "ConstantKernel(constant_value=1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)" }
```

Classifiers have `output_shape` `null` (the classes come from the data), `SVR`/`SVC` have
`num_params` `null`. See [graph-format.md](graph-format.md#classical-models) for the rules
and the catalogue fields `family`, `output_kind` and `input_slots` of `GET /api/layers/`.

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

For a classical model the code is a scikit-learn script (`make_model()`, `--csv`, `--save
model.joblib`, `--predict`); `class_name` does not apply, and without `include_training`
it keeps `--predict` to use a downloaded `.joblib` file.

```bash
curl -X POST http://localhost:8080/api/generate/?download=1 \
  -H "Content-Type: application/json" -d @my_network.json -o model.py
```

## Accounts (JWT)

| Method | Path                       | Body                               | Response                    |
| ------ | -------------------------- | ---------------------------------- | --------------------------- |
| POST   | `/api/auth/register/`      | `{username, email, password, accept_terms?}` | `201 {user, access, refresh}` |
| POST   | `/api/auth/token/`         | `{username, password}`             | `{access, refresh}`         |
| POST   | `/api/auth/token/refresh/` | `{refresh}`                        | `{access, refresh}` (rotated) |
| POST   | `/api/auth/logout/`        | `{refresh}`                        | blacklists the refresh token |
| GET    | `/api/auth/me/`            | –                                  | `{id, username, email, date_joined}` |

Send the access token as `Authorization: Bearer <access>`. Access tokens live 30 minutes,
refresh tokens 7 days. The e-mail address is required for new accounts and can be used by one
account only. Registration can be disabled with `BRICKDNN_ALLOW_REGISTRATION=false`. If the
server has terms of use (`GET /api/legal/` says `accept_terms_required`), registration needs
`accept_terms: true`; the acceptance is stored with the version of the terms.

### The own account

These endpoints need a website login (JWT), not an API token. Changes, the export and the
deletion share a rate limit (`BRICKDNN_ACCOUNT_RATE`, 30/hour).

| Method | Path                          | Body                                  | Response                    |
| ------ | ----------------------------- | ------------------------------------- | --------------------------- |
| GET    | `/api/auth/account/`          | –                                     | `Account`                   |
| PATCH  | `/api/auth/account/`          | `{email}`                             | `Account`                   |
| POST   | `/api/auth/account/password/` | `{current_password, new_password}`    | `{access, refresh}`; all other logins end |
| GET    | `/api/auth/account/export/`   | –                                     | a ZIP file with everything stored about the account |
| POST   | `/api/auth/account/delete/`   | `{password, username}`                | `204`: the account and everything it owns are deleted |

`Account` is `{id, username, email, date_joined, last_login, last_active_at,
terms_accepted_at, terms_version, privacy_version, stored}`; `stored` counts the projects,
memberships, comments, data sources and preparations, datasets, training runs, API tokens and
connected apps, and the bytes of uploaded files. What the export holds and what deleting an
account deletes: [privacy.md](privacy.md).

### Legal pages (public)

| Method | Path                    | Response                                                       |
| ------ | ----------------------- | -------------------------------------------------------------- |
| GET    | `/api/legal/`           | `{pages: {imprint, privacy, terms}, terms_version, privacy_version, accept_terms_required, retention: {unused_uploads_days, inactive_accounts_days}}` |
| GET    | `/api/legal/<page>/`    | `{page, title, content, version, updated_at}`; `404` if the page is not set up |

`<page>` is `imprint`, `privacy` or `terms`; `content` is the operator's Markdown file and
`version` the start of its SHA-256 hash. See [privacy.md](privacy.md).

## API tokens

For scripts and AI assistants: personal tokens that do not expire after 30 minutes. Send them
like an access token, `Authorization: Bearer bdnn_…`; requests then act as the token's owner.
Read-only tokens may only use `GET` and the `POST` endpoints that change nothing (analyze,
generate, training check, predictions, MCP). The MCP server for AI assistants is described in
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
| GET / POST    | `/api/projects/<id>/versions/`         | History, newest first / name the current state `{label}` (editors) |
| GET / PATCH   | `/api/projects/<id>/versions/<v>/`     | One version with its graph / rename it `{label}` (editors) |
| POST          | `/api/projects/<id>/versions/<v>/restore/` | Make it the current state (editors)              |
| GET / POST    | `/api/projects/<id>/comments/`         | Comments (`?node=<layer id>`) / comment `{text, node_id?, parent?}` (everyone with access) |
| PATCH / DELETE| `/api/projects/<id>/comments/<c>/`     | `{text}` (author), `{resolved}` (author, editors, owner) / delete (author, owner) |
| GET / POST    | `/api/projects/<id>/data/`             | Datasets and preparations shared in the project / add your own `{kind: "dataset" or "prep", id}` (owner, editors) |
| DELETE        | `/api/projects/<id>/data/<kind>/<item>/` | Stop sharing it there (its owner or the project's owner) |
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

**History.** Every change keeps a copy of the new state. Changes by the same person within ten
minutes update the latest automatic entry; at most 200 automatic entries are kept, named ones
for good. A version is `{id, version, kind: "auto" | "named" | "restore", label, name,
num_layers, created_by, created_at, updated_at}` (plus `description` and `graph` when fetched
alone). Restoring is a normal change, so it is undone by restoring another version.

**Comments.** `{id, parent, node_id, text, author: {id, username}, resolved, resolved_by,
created_at, updated_at}`. Replies (`parent`) belong to the layer of their thread; deleting a
comment deletes its replies.

**Shared data and runs.** Members see datasets shared in their projects in
`GET /api/training/datasets/` (with `owner`, `mine` and `shared_in`) and can train on them;
editors of the project can change shared data preparations (not their source) and export
them. A training started with `"project": <id>` (owner or editor of it) belongs to the
project: members see it (`GET /api/training/runs/?project=<id>`, with `project` and
`started_by`), who started it and the project's editors can stop it, who started it and the
project's owner can delete it.

### Live editing (WebSocket)

`ws(s)://<host>/api/ws/projects/<id>/` lets everyone with access work on a project at the same
time. Send `{"type": "auth", "token": "<JWT access token>"}` first; the server answers
`{"type": "hello", project: {id, name, description, graph, version}, role, client_id, user}`.
Then:

| Client sends | |
| --- | --- |
| `{"type": "ops", "id": "<your message id>", "ops": [...]}` | Changes (owner and editors) |
| `{"type": "presence", "state": {...}}` | E.g. selected layers or a layer being dragged; passed on, not stored |
| `{"type": "ping"}` | Answered with `pong` |

Ops: `add_node {node}`, `update_node {id, set: {params?, position?, name?, type?}}` (params are
merged key by key, `null` removes a key), `remove_node {id}` (with its connections),
`add_edge {source, target}`, `remove_edge {source, target}`, `set_meta {name?, description?}`.
The server applies them in order to the stored graph and sends
`{"type": "ops", ops, version, by, client_id, id}` to everyone, the sender included. Ops that
cannot be applied (e.g. a change to a layer someone deleted) are left out; if none could be
applied, the sender gets `rejected`.

Other messages from the server: `joined` / `left` (who is here), `presence`, `replaced` (the
whole project changed: saved through the REST API or MCP, or a version was restored),
`comment` (`action`: created, updated or deleted), `history`, `access` (your role changed;
`null`: removed), `deleted`, `error`. Close codes: 4401 not logged in, 4403 access removed,
4404 no such project (or no access), 4429 too many messages.

## Training (login required)

Training runs on the server in a separate process per run, see [training.md](training.md).
`GET /api/training/` is public so that the UI can show whether training is available.

| Method | Path                                   | Description                                         |
| ------ | -------------------------------------- | --------------------------------------------------- |
| GET    | `/api/training/`                       | Availability, devices, built-in datasets (`train_size` training and `test_size` official test images), limits, `classical` |
| GET    | `/api/training/datasets/`              | Your uploaded and prepared datasets                 |
| POST   | `/api/training/datasets/`              | Upload `file` (`.csv` or `.zip`), optional `name` (multipart) |
| GET / DELETE | `/api/training/datasets/<id>/`   | One dataset / delete it with its files              |
| POST   | `/api/training/check/`                 | `{graph, dataset, config?}`: does the network fit the data? |
| GET    | `/api/training/runs/`                  | Your runs, newest first                             |
| POST   | `/api/training/runs/`                  | Start a run `{name, graph, dataset, config, tuning?}` |
| GET    | `/api/training/runs/<id>/`             | Run with live state (history, progress, log)        |
| POST   | `/api/training/runs/<id>/stop/`        | Stop after the current batch (the best weights so far are kept) |
| DELETE | `/api/training/runs/<id>/`             | Delete the run (stops it first)                     |
| GET    | `/api/training/runs/<id>/weights/`     | Download the trained `state_dict` (`.pt`) or fitted classical model (`.joblib`); with `?sig=…` from a download link without login |
| POST   | `/api/training/runs/<id>/predict/`     | Run the trained network on a few inputs `{inputs, input_scale?, top_k?}` |

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

`POST /api/training/check/` answers `{"ok", "errors", "warnings", "plan", "fixes"}`: `plan` is
how the data is fed and which loss is used (`task`, `inputs`, `targets`, `image`,
`class_names`, `notes`), `fixes` the parameter changes that make the network fit the data,
applied together (empty when it fits or when other layers would have to change):

```json
"fixes": [
  {"node": "input", "type": "Input", "params": {"shape": [4]}, "previous": {"shape": [784]},
   "summary": "Input shape 784 → 4"},
  {"node": "fc3", "type": "Linear", "params": {"out_features": 3}, "previous": {"out_features": 10},
   "summary": "Outputs of the last layer 10 → 3"}
]
```

Only the Input layer (`shape`, `dtype`) and the last Linear layer (`out_features`) are changed;
a fix for surplus class outputs can also come with `"ok": true`. See
[training.md](training.md#adapting-the-network-to-the-data).

The check also takes the `config` (optional) to check what depends on it, e.g. `max_samples`
against the row limit of a classical model. For classical models, only the Input layer's
`shape` is ever suggested (their output follows the data).

`POST /api/training/runs/` answers incompatible combinations with explanations, e.g.
`400 {"detail": "...", "errors": ["The network outputs 5 values, but MNIST has 10 classes. ..."],
"warnings": []}`. Starting a run while another one is active gives `409`, and `503` when
training is disabled or the library the network needs is missing (PyTorch for neural
networks, scikit-learn for classical models).

**Classical models** (Gaussian processes, SVMs, see [training.md](training.md#classical-models))
train with scikit-learn and do not need PyTorch. `GET /api/training/` describes them separately,
because `available` and `reason` are about PyTorch:

```json
"classical": {
  "available": true, "reason": null, "sklearn_version": "1.9.1",
  "max_rows": { "gaussian_process": 5000, "svm": 50000 },
  "tuning": false
}
```

Only `val_split`, `test_split`, `max_samples` (at most `max_rows`) and `seed` of `config` apply;
a `tuning` gives `400`. Their runs have `family` `"classical"` (otherwise `"neural"`) and
`epochs` 1.

A run's `status` is `starting`, `preparing` (loading or downloading data), `running`,
`completed`, `stopped` or `failed`. `GET /api/training/runs/<id>/` includes `state` with the
per-epoch `history` (`train_loss`, `val_loss`, `train_accuracy`, `val_accuracy`, `val_rmse`),
the current `progress`, the `best` epoch, `weights_epoch` (the epoch whose weights are tested
and saved, set when the epochs are over), `stopped_early`, the `test` result, `preprocessing`
and the last log lines. Poll it about once per second while a run is active. `state.family`
is `neural` or `classical`; a classical run is fitted once: it has no `progress`, a single
`history` entry when the fit is done (regression adds `val_r2`), `test` adds `r2`, and
`state.classical` holds `estimator`, `label`, `max_rows`, `rows` (fitted on), `subsampled_from`,
`fit_seconds`, `initial_kernel`, `kernel` (fitted), `log_marginal_likelihood`,
`support_vectors` and `train` (the metrics on the training rows), see
[training.md](training.md#classical-models).

Every run also has `metrics` (the last epoch while it runs, then the kept epoch, plus
`test_loss`, `test_accuracy`, `test_rmse`; `val_r2` and `test_r2` for classical regression,
`null` otherwise), `family`, `tuning` (the normalised search settings or
`null`), `trials_done` and `best_trial` (`{trial, params, val_loss, val_accuracy, val_rmse,
weights_epoch}`). In a search, `state.trials` lists every trial with its `params`, `status`,
the metrics of its kept weights, `weights_epoch` and `stopped_early`; `history`, `best` and
`metrics` belong to the running trial and, at the end, to the best one. The file format is
documented in `backend/brickdnn/training/state.py`.

**Download links.** `GET /api/training/runs/<id>/weights/?sig=…` works without login: the
MCP tool `get_weights_download` creates such links. They are valid for 15 minutes, signed for
one run and one user, and refused (`403`, with the reason in `detail`) when they have expired,
were changed or belong to another run, or when that user can no longer see the run. Without
`sig`, the endpoint needs a login as before.

**Predictions.** `POST /api/training/runs/<id>/predict/` runs the trained network of a finished
or stopped run (everyone who can see the run, also with a read-only token):

```json
{ "inputs": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, [6.7, 3.0, 5.2, 2.3]], "top_k": 2 }
```

```json
{
  "run": { "id": "…", "name": "Iris net" },
  "task": "classification",
  "predictions": [
    { "class": "setosa", "index": 0, "probability": 0.97, "top": [{ "class": "setosa", "index": 0, "probability": 0.97 }, { "class": "versicolor", "index": 1, "probability": 0.02 }] },
    { "class": "virginica", "index": 2, "probability": 0.88, "top": [...] }
  ]
}
```

At most 64 samples and 250,000 numbers. Networks trained on a CSV file take one row per sample,
as an object `{column: value}` or a list in column order, with the raw values of the CSV file
(`input_scale` `raw`, the default) or already standardised ones (`normalized`); regression
answers `{"value": …}` per sample in the units of the target column (named in `target`). Image
networks take one image per sample as a nested array of channels × height × width (grayscale
also height × width) or in the shape of the Input layer, and need `input_scale`: `raw`
(pixels 0–255), `unit` (0–1) or `normalized` (normalised like in training, see
`state.preprocessing`). Runs on demo data take
arrays of the Input shape as they are (and add a `note`: their predictions mean nothing).
`top_k` (1–10, default 3) is the number of classes in `top`.

Classical models take the same rows. Classifiers give their own class probabilities (an SVC
calibrated with Platt scaling, a Gaussian process its predictive probabilities); a Gaussian
process regressor adds `"std"`, its predictive standard deviation in the units of the target,
to every prediction (`{"value": 101.3, "std": 0.4}`), and the answer a `std_note`.

The inputs are checked and prepared like the training data from the run's
`state.preprocessing`; the network then runs in a separate process (CPU, at most 30 seconds,
`BRICKDNN_MAX_PREDICTIONS` at a time, default 1). Answers: `400 {"detail", "inputs"}` when the
inputs do not fit (`inputs` describes what the network takes), `409` when the run has no
weights or its preprocessing is unknown (e.g. it failed early), `503` when training is
disabled, PyTorch (scikit-learn for classical models) is missing or other predictions are
running (`Retry-After: 5`), `500` when the
prediction itself fails or takes too long, `429` above the rate limit.

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
preparations: 60/hour, predictions: 120/hour, previews and downloads of preparations:
1200/hour, MCP requests: 600/hour, OAuth registration, token and revocation requests:
120/minute, changes to the own account (e-mail, password, export, deletion): 30/hour. Adjust
with `BRICKDNN_ANON_RATE`, `BRICKDNN_USER_RATE`, `BRICKDNN_AUTH_RATE`,
`BRICKDNN_TRAINING_RATE`, `BRICKDNN_UPLOAD_RATE`, `BRICKDNN_PREDICT_RATE`,
`BRICKDNN_DATA_RATE`, `BRICKDNN_MCP_RATE`, `BRICKDNN_OAUTH_RATE`, `BRICKDNN_ACCOUNT_RATE` and
`BRICKDNN_SHARING_RATE` (invitations and link changes, 120/hour)
(e.g. `100/minute`).
