# REST API

All endpoints live under `/api/`. Requests and responses are JSON. Code generation does not
need an account; saved networks do.

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

## Rate limits

Anonymous clients: 300 requests/minute, logged-in users: 600/minute, login and registration:
20/minute. Adjust with `BRICKDNN_ANON_RATE`, `BRICKDNN_USER_RATE` and `BRICKDNN_AUTH_RATE`
(e.g. `100/minute`).
