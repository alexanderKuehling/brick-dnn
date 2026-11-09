# MCP server, OAuth and API tokens

Brick DNN has a built-in [MCP](https://modelcontextprotocol.io) server, so AI assistants such
as Claude can build networks (and classical models: Gaussian processes, SVMs), check them,
generate PyTorch or scikit-learn code, save them in your account, train them on your server,
predict with the trained networks and hand you their weights. The
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

- **Full access** can also save and delete networks, upload and delete datasets and start,
  stop or delete trainings. **Read only** can look at everything (layers, examples, your
  networks and training runs), analyse networks, generate code, predict with trained networks
  and download their weights, but change nothing.
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
| `list_layer_types`      | read   |             | All layer types with their parameters, kernel bricks and classical models |
| `list_examples`         | read   |             | Example networks with their graphs                              |
| `analyze_network`       | read   |             | Validate a graph, infer shapes and parameter counts, explain errors |
| `generate_pytorch_code` | read   |             | PyTorch script for a valid graph (scikit-learn for a classical model) |
| `list_projects`         | read   |             | Your saved networks and the ones shared with you                |
| `get_project`           | read   |             | One saved network with its graph                                |
| `save_project`          | full   | yes         | Create a network or update one (owner, editors)                 |
| `delete_project`        | full   | yes         | Delete a network (owner only)                                   |
| `get_training_options`  | read   |             | Whether training is available, datasets, default settings, search, upload limit |
| `upload_dataset`        | full   |             | Store a small CSV file (sent as text) as a training dataset     |
| `delete_dataset`        | full   | yes         | Delete one of your own datasets                                 |
| `check_training`        | read   |             | Does a network fit a dataset (optionally with the settings)? With `fixes` for the Input and last layer |
| `start_training`        | full   |             | Train in the background                                         |
| `list_training_runs`    | read   |             | Your training runs with status and metrics (regressions: RMSE and R²) |
| `get_training_run`      | read   |             | One run with its metrics per epoch                              |
| `stop_training`         | full   |             | Stop a run after the current batch (starter, project editors)   |
| `delete_training_run`   | full   | yes         | Delete a run with its weights (starter, project owner)          |
| `get_weights_download`  | read   |             | Short-lived download link for the trained weights, with model code and preprocessing |
| `predict`               | read   |             | Run a trained network on up to 64 samples                       |

Read-only tokens and apps allowed read-only access only see the read tools. Networks use the [graph format](graph-format.md).

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

`upload_dataset {name, csv, project_id?}` takes the content of a CSV file as text: features in
all columns but the last, the target in the last one (see [training.md](training.md)). It is
stored and checked exactly like an upload in the web app and returned like
`get_training_options` lists datasets, plus `reference` (`{"kind": "upload", "id": …}`) to
train on it. With `project_id` (a project you own or edit) it is shared there right away.
CSV text can have up to `BRICKDNN_MCP_UPLOAD_MB` (default 5 MB, at most
`BRICKDNN_MAX_UPLOAD_MB`); the MCP endpoint accepts requests of about twice that size, while
other JSON requests stay limited to 1 MB. The bundled nginx lets requests to `/api/mcp` through
up to 12 MB.
Larger files belong in the web app.

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
