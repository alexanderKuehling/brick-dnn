# REST API

All endpoints live under `/api/`. Requests and responses are JSON. Code generation does not
need an account; saved networks and training do.

## Code generation (public)

| Method | Path              | Description                                                   |
| ------ | ----------------- | ------------------------------------------------------------- |
| GET    | `/api/`           | Index of endpoints and version                                |
| GET    | `/api/health/`    | `{"status": "ok", "version": "1.0.0"}`                        |
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
  "inputs": [],
  "outputs": [],
  "total_params": null
}
```

`node` is `null` for errors about the whole graph. `params` contains the resolved values,
including inferred `auto` parameters.

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
| POST   | `/api/auth/register/`      | `{username, password, email?}`     | `201 {user, access, refresh}` |
| POST   | `/api/auth/token/`         | `{username, password}`             | `{access, refresh}`         |
| POST   | `/api/auth/token/refresh/` | `{refresh}`                        | `{access, refresh}` (rotated) |
| POST   | `/api/auth/logout/`        | `{refresh}`                        | blacklists the refresh token |
| GET    | `/api/auth/me/`            | –                                  | `{id, username, email, date_joined}` |

Send the access token as `Authorization: Bearer <access>`. Access tokens live 30 minutes,
refresh tokens 7 days. Registration can be disabled with `BRICKDNN_ALLOW_REGISTRATION=false`.

## Saved networks (login required)

| Method        | Path                   | Description                              |
| ------------- | ---------------------- | ---------------------------------------- |
| GET           | `/api/projects/`       | Your networks, newest first              |
| POST          | `/api/projects/`       | `{name, graph, description?}`            |
| GET           | `/api/projects/<id>/`  | One network                              |
| PUT / PATCH   | `/api/projects/<id>/`  | Update                                   |
| DELETE        | `/api/projects/<id>/`  | Delete                                   |

Every user only sees their own networks. Graphs may be saved with errors (work in progress)
but are limited to 500 layers and 2000 connections.

## Training (login required)

Training runs on the server in a separate process per run, see [training.md](training.md).
`GET /api/training/` is public so that the UI can show whether training is available.

| Method | Path                                   | Description                                         |
| ------ | -------------------------------------- | --------------------------------------------------- |
| GET    | `/api/training/`                       | Availability, devices, built-in datasets, limits    |
| GET    | `/api/training/datasets/`              | Your uploaded datasets                              |
| POST   | `/api/training/datasets/`              | Upload `file` (`.csv` or `.zip`), optional `name` (multipart) |
| GET / DELETE | `/api/training/datasets/<id>/`   | One dataset / delete it with its files              |
| POST   | `/api/training/check/`                 | `{graph, dataset}`: does the network fit the data?  |
| GET    | `/api/training/runs/`                  | Your runs, newest first                             |
| POST   | `/api/training/runs/`                  | Start a run `{name, graph, dataset, config, tuning?}` |
| GET    | `/api/training/runs/<id>/`             | Run with live state (history, progress, log)        |
| POST   | `/api/training/runs/<id>/stop/`        | Stop after the current batch (weights are kept)     |
| DELETE | `/api/training/runs/<id>/`             | Delete the run (stops it first)                     |
| GET    | `/api/training/runs/<id>/weights/`     | Download the trained `state_dict` (`.pt`)           |

`dataset` is one of `{"kind": "random"}` (demo data), `{"kind": "builtin", "id": "mnist"}`
(`mnist`, `fashion_mnist`, `cifar10`) or `{"kind": "upload", "id": 3}`. `config` may contain
`epochs`, `batch_size`, `learning_rate`, `optimizer` (`adam`, `adamw`, `sgd`), `weight_decay`,
`val_split`, `test_split` (0 = no test set; validation and test together at most 0.8),
`device` (`auto`, `cpu`, `cuda`, `mps`), `max_samples` and `seed`; missing values use the
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
the current `progress`, the `best` epoch, the `test` result, `preprocessing` and the last log
lines. Poll it about once per second while a run is active.

Every run also has `metrics` (the last epoch plus `test_loss`, `test_accuracy`, `test_rmse`),
`tuning` (the normalised search settings or `null`), `trials_done` and `best_trial`
(`{trial, params, val_loss, val_accuracy, val_rmse}`). In a search, `state.trials` lists every
trial with its `params`, `status` and final metrics; `history`, `best` and `metrics` belong to
the running trial and, at the end, to the best one. The file format is documented in
`backend/brickdnn/training/state.py`.

## Rate limits

Anonymous clients: 300 requests/minute, logged-in users: 600/minute, login and registration:
20/minute, starting training runs: 120/hour, dataset uploads: 60/hour. Adjust with
`BRICKDNN_ANON_RATE`, `BRICKDNN_USER_RATE`, `BRICKDNN_AUTH_RATE`, `BRICKDNN_TRAINING_RATE` and
`BRICKDNN_UPLOAD_RATE` (e.g. `100/minute`).
