# REST API

All endpoints live under `/api/`. Requests and responses are JSON. Code generation does not
need an account; saved networks, training and data preparation do.

## Code generation (public)

| Method | Path              | Description                                                   |
| ------ | ----------------- | ------------------------------------------------------------- |
| GET    | `/api/`           | Index of endpoints and version                                |
| GET    | `/api/health/`    | `{"status": "ok", "version": "1.1.0", "checks": {...}}`, see below |
| GET    | `/api/layers/`    | `{"categories": [...], "layers": [LayerSpec, ...]}`           |
| GET    | `/api/examples/`  | Example networks `[{id, name, description, graph}]`           |
| POST   | `/api/analyze/`   | Validate a graph, infer shapes                                |
| POST   | `/api/generate/`  | Generate PyTorch code (scikit-learn for classical models)     |

### `GET /api/health/`

Whether the server works, cheap enough for frequent checks (Docker's health check, the uptime
workflow in [deploy/server](../deploy/server/README.md#monitoring)):

```json
{ "status": "degraded", "version": "1.1.0", "checks": { "database": "ok", "worker": "failed", "disk": "ok" } }
```

| Check      | Fails when                                                                 |
| ---------- | -------------------------------------------------------------------------- |
| `database` | `SELECT 1` fails. The answer is then `503` with `"status": "error"`.       |
| `worker`   | With the training queue: no training worker sent a heartbeat within `BRICKDNN_WORKER_TIMEOUT` seconds. |
| `disk`     | The data directory has less than `BRICKDNN_HEALTH_MIN_FREE_MB` free (default 1024; `0`: no check). |
| `backup`   | With `BRICKDNN_BACKUP_DIR`, once `manage.py backup` ran: the last backup failed, or the newest is older than `BRICKDNN_BACKUP_MAX_AGE_HOURS` (default 50). |

A failed check other than the database makes the status `degraded` (still `200`); checks a
server does not use are left out. Everyone sees only these names and `ok`/`failed`. Staff
accounts, and requests with the header `X-Health-Token: <BRICKDNN_HEALTH_TOKEN>`, also get a
`detail` object (free space, seconds since the last heartbeat, error messages).

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

Every node also has `output_kind` (`tensor`, `kernel` or `estimator`: what its outgoing
edges carry; classical models and ensembles pass on the model itself), `kernel` (kernel
bricks and Gaussian processes: the kernel as scikit-learn code, otherwise `null`) and `task`
(classical models and ensembles: `regression` or `classification`); ensembles also have
`members` (their models in the order of the edges: `[{"node", "name", "type", "estimator",
"label"}]`). The top level adds `family` (`neural` or `classical`) and `classical`
(`{"node", "type", "estimator", "task", "kernel", "members"}` for a network with a classical
model or an ensemble, else `null`; `members` is `null` for a single model). For example a
Gaussian process without a kernel brick:

```json
"gp": { "type": "GaussianProcessRegressor", "output_shape": [1], "dtype": "float", "num_params": 3,
        "params": { "alpha": 1e-10, "normalize_y": true, "n_restarts_optimizer": 0 },
        "output_kind": "estimator", "task": "regression",
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
| POST   | `/api/auth/token/`         | `{username, password}`             | `{access, refresh}`, or a two-factor challenge (below) |
| POST   | `/api/auth/token/refresh/` | `{refresh}`                        | `{access, refresh}` (rotated) |
| POST   | `/api/auth/logout/`        | `{refresh}`                        | blacklists the refresh token |
| GET    | `/api/auth/me/`            | –                                  | `{id, username, email, date_joined}` |

Send the access token as `Authorization: Bearer <access>`. Access tokens live 30 minutes,
refresh tokens 7 days. The e-mail address is required for new accounts and can be used by one
account only. Registration can be disabled with `BRICKDNN_ALLOW_REGISTRATION=false`. If the
server has terms of use (`GET /api/legal/` says `accept_terms_required`), registration needs
`accept_terms: true`; the acceptance is stored with the version of the terms. With e-mail set
up, sign-up sends a link to confirm the address.

### E-mail, password reset and two-factor authentication

See [email.md](email.md) for what is sent and when an address may receive reset links.

| Method | Path | Body | Response |
| ------ | ---- | ---- | -------- |
| POST | `/api/auth/token/two-factor/` | `{challenge, code}` | `{access, refresh}` (plus `recovery_codes_left` after a recovery code) |
| GET  | `/api/auth/password-reset/` | – | `{available}`: whether this server sends reset links |
| POST | `/api/auth/password-reset/` | `{login}` (username or e-mail) | `{detail}`, the same for every login; `503` without e-mail |
| POST | `/api/auth/password-reset/check/` | `{uid, token}` (from the link) | `{valid, username}` |
| POST | `/api/auth/password-reset/confirm/` | `{uid, token, new_password}` | `{detail, username}`; all logins of the account end; `400 {code: "invalid_link"}` |
| POST | `/api/auth/account/email/verify/` | `{token}` (from the link) | `{email, email_verified, …}`; only the account the link was sent to (`400 {code: "other_account" \| "other_address" \| "invalid_link"}`) |
| POST | `/api/auth/account/email/resend/` | – | `{detail}`; `503` without e-mail; `BRICKDNN_EMAIL_VERIFY_RATE` (5/hour) |
| GET  | `/api/auth/two-factor/` | – | `TwoFactorState` |
| POST | `/api/auth/two-factor/setup/` | `{password}` | `{secret, otpauth_uri}` (shown once; not on until confirmed) |
| POST | `/api/auth/two-factor/enable/` | `{code}` | `{recovery_codes, state, access, refresh}`; all other logins end |
| POST | `/api/auth/two-factor/disable/` | `{password, code}` | `TwoFactorState` |
| POST | `/api/auth/two-factor/recovery-codes/` | `{password, code}` | `{recovery_codes, state}`; the old codes stop working |

**Logging in with two-factor authentication.** When it is on, `POST /api/auth/token/` with the
right password answers `{two_factor_required: true, challenge, expires_in}` (no tokens). The
challenge is signed, valid for 5 minutes and ends with a password change. `POST
/api/auth/token/two-factor/` with it and a 6-digit code from the app, or a recovery code
(`xxxx-xxxx-xxxx-xxxx`), returns the tokens. Wrong codes: `400 {code: [...]}`; an expired
challenge: `400 {code: "challenge_expired"}`; after 5 wrong codes in 15 minutes: `429`. The
step is limited per IP address too (`BRICKDNN_TWO_FACTOR_RATE`, 30/hour). API tokens and
OAuth apps are not affected. `TwoFactorState` is `{enabled, enabled_at, recovery_codes_left}`.
The `/api/auth/two-factor/…` endpoints need a website login and share the account rate limit.

**Password reset** is limited per IP address (`BRICKDNN_PASSWORD_RESET_RATE`, 10/hour) and
to 3 e-mails an hour per account. The link is `/reset-password#uid=…&token=…` (the fragment
stays in the browser), valid for an hour and void once the password changed, the account
logged in or its address changed.

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

`Account` is `{id, username, email, email_verified, email_verified_at, email_sending,
password_reset_available, two_factor, date_joined, last_login, last_active_at,
terms_accepted_at, terms_version, privacy_version, stored}` (`email_sending`: the server sends
e-mail; `password_reset_available`: a forgotten password could be reset by e-mail;
`two_factor`: `TwoFactorState`). Changing the e-mail address makes it unconfirmed and sends a
link to the new address and a notice to the old one;` `stored` counts the projects,
memberships, comments, data sources and preparations, datasets, training runs, API tokens,
connected apps and entries of the agent activity log (`agent_actions`), and the bytes of
uploaded files. What the export holds and what deleting an
account deletes: [privacy.md](privacy.md).

### Paid plans (Paddle)

Only on a server that sells plans, see [billing.md](billing.md). Without it,
`GET /api/billing/` answers `{"enabled": false}` and every other path below `404`.

| Method | Path                            | Body              | Response                         |
| ------ | ------------------------------- | ----------------- | -------------------------------- |
| GET    | `/api/billing/`                 | –                 | `BillingInfo` (public)           |
| GET    | `/api/billing/subscription/`    | –                 | `BillingState`                   |
| POST   | `/api/billing/checkout/`        | `{plan}`          | `{plan, price_id, customer_id, custom_data}` for Paddle's checkout |
| POST   | `/api/billing/change-plan/`     | `{plan, preview?}` | `BillingState`; with `preview` `{plan, upgrade, now, next_billed_at}` and nothing changes |
| POST   | `/api/billing/cancel/`          | –                 | `BillingState`: cancelled to the end of the paid period |
| POST   | `/api/billing/reactivate/`      | –                 | `BillingState`: the cancellation is taken back |
| POST   | `/api/billing/portal/`          | –                 | `{url}`: Paddle's customer portal, logged in |
| POST   | `/api/billing/refresh/`         | –                 | `BillingState`, after asking Paddle |
| POST   | `/api/billing/paddle/webhook/`  | Paddle's event    | `{received, new, outcome}`; for Paddle only |

`BillingInfo` is `{enabled, provider: "paddle", environment: "sandbox"|"live", client_token,
currency, prices_include_tax, overage_billed, plans}`; a plan is a plan of
`GET /api/account/usage/` plus `purchasable` and `default`. `BillingState` is
`{plan, unlimited, subscription, can_subscribe, has_billing_profile, email_missing}` with
`subscription` `null` or `{status, plan, plan_name, renews_at, ends_at, ended_at,
current_period_end, payment_failed, grace_until, can_change_plan, can_cancel,
can_reactivate}`; `status` is Paddle's (`active`, `trialing`, `past_due`, `paused`,
`canceled`). `now` of a preview is `{action: "charge"|"credit", amount, currency}` or `null`.

All but the first and the last need a website login (JWT), not an API token; changes are
limited to `BRICKDNN_BILLING_RATE` (60/hour). `409` means that the request does not fit the
subscription (e.g. a checkout while subscribed, a plan change while a payment is open), `503`
that Paddle did not answer or refused: nothing was changed. The webhook accepts only requests
with a valid `Paddle-Signature` (`403` otherwise), at most 256 KB, and
`BRICKDNN_BILLING_WEBHOOK_RATE` (600/minute per IP address).

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

`ApiToken` is `{id, name, prefix, read_only, created_at, last_used_at, expires_at,
budget_cpu_minutes, used_cpu_minutes}`; `PATCH /api/auth/api-tokens/<id>/
{budget_cpu_minutes}` sets its monthly compute budget (see
[Usage and budgets](#usage-and-budgets)).
`expires_in_days` is `7`, `30`, `90`, `365` or `null` (never, the default). The token itself
is only returned by `POST`; only its hash is stored. At most 20 tokens per user. These three
endpoints need a website login (JWT): an API token cannot manage tokens. A new token gets the
account's default [approval rule](#approvals). A request with a token whose rule says that
it needs approval (trainings above its limit, sharing, deleting) is refused with `403`: the
REST API cannot ask.

## Agent activity

What AI agents and scripts did with the user's API tokens and connected apps (see
[mcp.md](mcp.md#activity-log)). Website login (JWT) only; each user sees their own entries.

| Method | Path                            | Response                                         |
| ------ | ------------------------------- | ------------------------------------------------ |
| GET    | `/api/agent-activity/`          | `{results: [AgentAction], has_more, latest_id, retention_days, approvals}`, newest first |
| GET    | `/api/agent-activity/summary/`  | `{agents, tools, total, latest_id, latest_at, retention_days}` |

Query parameters of the list: `limit` (default 50, at most 200), `before=<id>` (older
entries: the next page), `after=<id>` (only newer entries: poll with the last `latest_id`),
and the filters `agent=token:<id>` or `app:<connection id>`, `tool=<tool or route>`,
`outcome=ok|error|pending|approved|denied`, `source=mcp|api`, `writes=true`, `project=<id>` and `since=<ISO date or
time>`. `latest_id` is the user's newest entry whatever the filters. With `after`, `has_more`
means more new entries than `limit` arrived: load the first page again.

`AgentAction` is `{id, created_at, source ("mcp" | "api"), agent: {kind ("token" | "app"), id,
key, name, prefix}, user_agent, tool, method, writes, outcome ("ok" | "error" | "pending" |
"approved" | "denied"), error, status_code, duration_ms, summary, arguments, targets: [{kind,
id, name?}], project, approval}`; `approval` is the id of the [approval request](#approvals)
an entry belongs to (outcomes `pending`, `approved`, `denied`, or `error` when an approved
action failed).
`arguments` are summarised and redacted; `targets` kinds are `project`, `run`, `dataset`,
`data_source` and `data_prep`. The summary's `agents` are `{kind, id, key, name, prefix,
active, actions, errors, last_at, last_id}`, the most recently active first; `active` is false
for revoked tokens and disconnected apps.

## Approvals

Actions of agents that wait for the user's approval, and the rules per token and app (see
[mcp.md](mcp.md#approvals)). Website login (JWT) only: API tokens and apps cannot decide.

| Method | Path                                  | Body                    | Response |
| ------ | ------------------------------------- | ----------------------- | -------- |
| GET    | `/api/approvals/`                     | –                       | `{results: [ApprovalRequest], pending_count}`, newest first; `?status=pending\|decided`, `?limit=` (default 50, at most 200) |
| GET    | `/api/approvals/<id>/`                | –                       | `ApprovalRequest` |
| POST   | `/api/approvals/<id>/approve/`        | –                       | `ApprovalRequest`, done as the agent: `status` `executed` or `failed`; approving again changes nothing; `409 {detail, approval}` if denied or expired |
| POST   | `/api/approvals/<id>/deny/`           | `{message?}`            | `ApprovalRequest` (`denied`); `409` if approved or expired |
| GET    | `/api/approvals/rules/`               | –                       | `{default: Rule, builtin_default: Rule, agents: {"token:<id>" \| "app:<id>": Rule}, gated_tools, expiry_hours}` |
| PUT    | `/api/approvals/rules/default/`       | `Rule`                  | `Rule`: the default new tokens and apps get |
| PUT    | `/api/approvals/rules/token/<id>/`    | `Rule`                  | `Rule` of an API token |
| PUT    | `/api/approvals/rules/app/<id>/`      | `Rule`                  | `Rule` of a connected app |

`Rule` is `{training_cpu_minutes (null: never ask; 0: always), publishing, deleting}`.
`ApprovalRequest` is `{id, status ("pending" | "approved" | "executed" | "failed" | "denied" |
"expired"), category ("training" | "publishing" | "deleting"), tool, action, agent: {kind, id,
key, name, active}, summary, reason, estimate: {cpu_minutes, low_cpu_minutes,
high_cpu_minutes, cost, currency} | null, limit_cpu_minutes, network, dataset, exposes,
warning, targets, arguments (summarised as in the activity log), created_at, expires_at,
decided_at, finished_at, message, error, done}`. `GET /api/agent-activity/` also returns
`approvals: {pending_count, pending: [ApprovalRequest]}` with every poll.

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
| GET    | `/api/training/`                       | Availability, devices, built-in datasets (`train_size` training and `test_size` official test images; `task` is `classification` or `segmentation`), limits, `classical`, `segmentation`, `queue` |
| GET    | `/api/training/datasets/`              | Your uploaded and prepared datasets (`kind`: `csv`, `images` or `segmentation`) |
| POST   | `/api/training/datasets/`              | Upload `file` (`.csv` or `.zip`), optional `name` and `kind` (multipart) |
| GET / DELETE | `/api/training/datasets/<id>/`   | One dataset / delete it with its files              |
| POST   | `/api/training/check/`                 | `{graph, dataset, config?, tuning?}`: does the network fit the data, what will it cost, do the budgets allow it, how long would it wait? |
| GET    | `/api/training/runs/`                  | Your runs, newest first                             |
| POST   | `/api/training/runs/`                  | Start a run `{name, graph, dataset, config, tuning?}` |
| GET    | `/api/training/runs/<id>/`             | Run with live state (history, progress, log)        |
| POST   | `/api/training/runs/<id>/stop/`        | Stop after the current batch (the best weights so far are kept); cancels a run that is still `queued` |
| DELETE | `/api/training/runs/<id>/`             | Delete the run (stops it first)                     |
| GET    | `/api/training/runs/<id>/weights/`     | Download the trained `state_dict` (`.pt`) or fitted classical model (`.joblib`); with `?sig=…` from a download link without login |
| POST   | `/api/training/runs/<id>/predict/`     | Run the trained network on a few inputs `{inputs, input_format?, input_scale?, top_k?}` or image files `{image, mask?}` |
| GET    | `/api/training/runs/<id>/predict-form/` | What the prediction panel needs: input columns (prepared and raw) with sample rows, or the image size; `available`/`reason` |
| POST   | `/api/training/runs/<id>/decision-threshold/` | Binary classification: the threshold its predictions use `{threshold}` (0–1; only who started the run) |
| GET    | `/api/training/runs/<id>/previews/`    | Segmentation runs: validation images with their mask and the predicted mask (`404` without) |
| GET    | `/api/training/runs/<id>/previews/<name>` | One of those pictures as a PNG file           |
| GET    | `/api/training/runs/<id>/explain/`     | Which explanations work for the run, and why the others do not |
| POST   | `/api/training/runs/<id>/explain/`     | Explain the trained model `{method, sample? \| input?, input_scale?, target?, layer?, node?}` |

`dataset` is one of `{"kind": "random"}` (demo data), `{"kind": "builtin", "id": "mnist"}`
(`mnist`, `fashion_mnist`, `cifar10`, or `shapes` for segmentation) or
`{"kind": "upload", "id": 3}`. `config` may contain
`epochs`, `batch_size`, `learning_rate`, `optimizer` (`adam`, `adamw`, `sgd`), `weight_decay`,
`val_split` (0–0.5; 0 = no validation data: the last epoch is kept, and `patience` and `tuning`
are refused), `test_split` (0 = no test set; validation and test together at most 0.8; built-in
datasets take `val_split` of their training images and always test on their official test
images, `test_split` is not used for them),
`device` (`auto`, `cpu`, `cuda`, `mps`), `max_samples`, `seed`, `patience` (early stopping:
stop when the validation loss has not improved for this many epochs, 1–1000; `null`, the
default, trains every epoch), `restore_best` (default `true`: test and save the weights of
the epoch with the lowest validation loss instead of the last one) and, for a segmentation,
`loss` (`cross_entropy`, the default, or `cross_entropy_dice`); missing values use the
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

Only the Input layer (`shape`, `dtype`) and the last Linear layer (`out_features`) are changed
(for a segmentation dataset the last `Conv2d` or `ConvTranspose2d` layer: `out_channels`);
a fix for surplus class outputs can also come with `"ok": true`. See
[training.md](training.md#adapting-the-network-to-the-data).

**Segmentation** ([training.md](training.md#image-segmentation)). A `.zip` upload with a folder
of images and a folder of masks becomes a dataset of `kind` `segmentation`; the form field
`kind` (`segmentation` or `images`) says which one is meant when the folders should not
decide. Its `info` has `num_samples` (images with a mask), `sample_shape` (channels, height
and width of the most common image size), `class_names`, `predefined_split` (with `train`,
`val` and `test` folders) and `segmentation`: `mask_format` (`binary`, `indices`, `levels` or
`colors`), `ignore_index` (255 or null), `class_pixel_share` (one share per class),
`image_sizes`, `unmatched_images`, `unmatched_masks`, `unmatched_examples` and
`mask_size_mismatches`. `GET /api/training/` lists the limits under `segmentation`
(`max_classes`, `max_images`, `max_input_side`, `batch_memory_mb`, `losses`). The plan of a
check has `task` `segmentation`, the target `{"kind": "mask", "num_classes", "binary"}` and
`segmentation` (`binary`, `ignore_index`, and `fixed_bytes` and `sample_bytes`: a training
step needs about `fixed_bytes + batch_size · sample_bytes` of memory; a `config` whose batch
does not fit into `batch_memory_mb` is an error that names the batch size that does).

A completed run with a single class or value output (classification, regression; neural or
classical) has `state.evaluation`: the kept weights measured once more on the validation and
the test rows ([training.md](training.md#metrics-of-the-finished-model) explains the metrics;
`backend/brickdnn/training/state.py` has every key). It is `null` or missing for other runs
and for runs of older versions. A run without validation rows (`val_split: 0`, or a dataset
from the Data tab without them) has `"val": null` and `null` validation values in its history
and metrics; with neither validation nor test rows, `"training"` holds the same metrics on the
training rows (optimistic: the model has seen them), otherwise it is `null`:

```json
"evaluation": {
  "task": "classification", "epoch": 7, "trial": null,
  "train": {"loss": 0.21, "accuracy": 0.93},
  "class_names": ["cat", "dog", "bird"], "matrix_classes_max": 50,
  "val": {"size": 1200, "loss": 0.25, "accuracy": 0.93, "balanced_accuracy": 0.91,
          "macro_f1": 0.9, "weighted_f1": 0.93, "log_loss": 0.25, "top_k_accuracy": null,
          "per_class": {"precision": [0.95, 0.9, null], "recall": […], "f1": […],
                        "support": [400, 420, 380]},
          "confusion_matrix": [[380, 15, 5], […], […]],
          "most_confused": [{"true": 1, "predicted": 0, "count": 15}]},
  "test": {…}
}
```

`confusion_matrix` counts rows (true class) against columns (predicted class); it is `null`
above `matrix_classes_max` classes, `top_k_accuracy` (`{"3": …, "5": …}`) only has k smaller
than the number of classes, and a per-class value is `null` where it is undefined. A
regression split has `size`, `loss`, `mae`, `mse`, `rmse`, `r2`, `mape` (a share; `null` when a
target is near 0), `median_absolute_error`, `max_error`, `residuals` (`q05`, `q25`, `median`,
`q75`, `q95` of predicted − actual) and `sample` (`actual` and `predicted`, at most 500 rows),
in the units of the target.

A **binary classification** (two classes) also has a `threshold_curve` in every split and
`binary` in the evaluation ([training.md](training.md#metrics-of-the-finished-model) explains
them). A row is predicted as the positive class (`positive_class`, the second one) when its
probability of that class is at least the threshold. The curves have the same thresholds in
every split (at most 199, ascending), chosen from the split named in `binary.source` (`val`,
else `test`, else `training`), which the suggestions are measured on too:

```json
"val": {…,
  "threshold_curve": {"positive_class": 1, "positives": 20, "negatives": 1180,
                      "roc_auc": 0.95, "pr_auc": 0.71,
                      "points": {"threshold": [0.0, 0.01, …], "precision": […], "recall": […],
                                 "fpr": […], "tp": […], "fp": […], "tn": […], "fn": […]}}},
"binary": {"positive_class": 1, "recall_target": 0.8, "source": "val",
           "best_f1": {"threshold": 0.31, "precision": 0.8, "recall": 0.7, "f1": 0.75,
                       "fpr": 0.003, "tp": 14, "fp": 4, "tn": 1176, "fn": 6},
           "recall": {…}}
```

`pr_auc` is the average precision; `roc_auc` is `null` without rows of both classes, `pr_auc`
and the suggestions without positive rows, a precision where nothing is predicted positive.
`best_f1` is the threshold with the best F1 score, `recall` the highest threshold whose recall
is at least `recall_target`.

**The decision threshold.** Runs have `decision_threshold`: for a binary classification the
threshold its predictions use (default `0.5`, the more probable class), `null` for every other
run. Who started the run changes it with `POST /api/training/runs/<id>/decision-threshold/`
`{"threshold": 0.12}` (a number from 0 to 1) and gets `{"id": "…", "decision_threshold":
0.12}`; others get `403` (also read-only tokens), other runs and values `400`.

Runs of a segmentation have `val_iou`, `val_dice` and `val_pixel_accuracy` in every entry of
`state.history` and in `metrics` (there also `test_iou`, `test_dice`, `test_pixel_accuracy`);
the accuracies are null. Once the weights are saved, `state.segmentation` has the results of
the kept weights per class and the previews:

```json
"segmentation": {
  "loss": "cross_entropy", "binary": false, "ignore_index": null,
  "class_names": ["background", "square", "circle", "triangle"],
  "colors": ["#1a1a19", "#3987e5", "#d95926", "#199e70"],
  "validation": {"epoch": 7, "iou": 0.79, "dice": 0.88, "pixel_accuracy": 0.97,
                 "class_iou": [0.98, 0.8, 0.82, 0.56], "class_dice": [0.99, 0.89, 0.9, 0.72]},
  "test": {"iou": 0.78, "dice": 0.87, "pixel_accuracy": 0.97, "class_iou": […], "class_dice": […]},
  "previews": {"count": 8, "epoch": 7, "split": "validation"}
}
```

`GET /api/training/runs/<id>/previews/` returns those previews (everyone who can see the run):

```json
{"epoch": 7, "split": "validation", "width": 64, "height": 64, "input_size": [64, 64],
 "class_names": ["background", "square", "circle", "triangle"], "colors": ["#1a1a19", …],
 "ignore_color": "#8a8f98",
 "samples": [{"index": 0, "iou": 0.92, "pixel_accuracy": 0.97, "classes": [0, 1, 3],
              "image": {"file": "0_image.png", "url": "data:image/png;base64,…"},
              "truth": {…}, "prediction": {…}, "overlay": {…}}]}
```

`image` is the picture as the network got it, `truth` its mask and `prediction` the predicted
mask (palette PNGs whose pixel values are the class indices, shown in `colors`), `overlay` the
prediction drawn over the image; `iou` is the mean IoU of that image over its `classes`.
`?embed=0` leaves the data URLs out; `GET …/previews/0_overlay.png` returns one file.

The check also has `estimate` and `budget` (pass `config` and `tuning` as for the start):

```json
"estimate": {"cpu_minutes": 2.1, "low_cpu_minutes": 0.7, "high_cpu_minutes": 6.3,
             "cost": 0.02, "currency": "EUR", "basis": "48,000 training and …",
             "note": "A rough estimate …", "details": {"train_samples": 48000, …}},
"budget": {"allowed": true, "reason": null, "warning": null, "run_limit_cpu_minutes": 15,
           "limit_reason": "run_limit", "remaining_cpu_minutes": 27.5,
           "resets_at": "2026-12-01T00:00:00Z"}
```

`estimate` is a rough estimate of the CPU time (null if the network does not fit; `cost` only
with a price), `budget.allowed` false with `reason` and `code` (`budget_exceeded`,
`storage_quota_exceeded`, `estimate_too_high`) if a start would be refused, `warning` if the
run will probably be stopped by its limit, `run_limit_cpu_minutes` the most it may use. A start
refused by a budget answers `403 {detail, code, …}`; the plan's limit of trainings at a time
answers `409` (with a training queue: its limit of waiting trainings). Runs have `usage: {cpu_minutes, limit_cpu_minutes, estimated_cpu_minutes,
stop_reason}` (`stop_reason`: `user`, `budget` or `run_limit`) and `started_via` (`{source:
"token" | "app", name}`, null in the web app); the answer to a start adds `budget_warning`.
See [training.md](training.md#usage-and-budgets).

## Usage and budgets

| Method | Path                          | Body                                  | Response                    |
| ------ | ----------------------------- | ------------------------------------- | --------------------------- |
| GET    | `/api/account/usage/`         | –                                     | `Usage` (any login, also an API token) |
| PATCH  | `/api/account/usage/`         | `{budget_cpu_minutes?, overage_enabled?, overage_cap_cpu_hours?}` | `Usage` (website login only) |
| PATCH  | `/api/auth/api-tokens/<id>/`  | `{budget_cpu_minutes}`                | `ApiToken` (website login only) |
| PATCH  | `/api/auth/connections/<id>/` | `{budget_cpu_minutes}`                | the connection (website login only) |

`Usage` is this calendar month's (UTC) usage, in CPU minutes and bytes:

```json
{"period": {"start": "2026-11-01", "resets_at": "2026-12-01T00:00:00Z"},
 "plan": {"key": "free", "name": "Free", "price": 0, "currency": "EUR",
          "cpu_hours_per_month": 0.5, "storage_gb": 1, "cpu_minutes_per_run": 15,
          "concurrent_trainings": 1, "priority": 0, "overage_price_per_cpu_hour": null},
 "unlimited": false,
 "plans": [ …the plans with prices, if the server has several… ],
 "upgrade_url": null,
 "overage": {"available": false, "enabled": false, "price_per_cpu_hour": null,
             "currency": "EUR", "cap_cpu_hours": null, "used_cpu_minutes": 0, "cost": null},
 "compute": {"unit": "cpu_minutes", "used": 12.5, "limit": 30, "remaining": 17.5,
             "available": 7.5, "allowance": 30, "own_budget": null, "run_limit": 15,
             "reserved_by_active_runs": 0,
             "by_kind": {"training": 12, "prediction": 0.5},
             "by_source": [{"source": "token", "id": 3, "name": "agent", "deleted": false,
                            "used_cpu_minutes": 10, "budget_cpu_minutes": 10, "cost": null}]},
 "storage": {"used_bytes": 1048576, "limit_bytes": 1073741824, "remaining_bytes": …,
             "by_kind": {"data_sources": 0, "datasets": 1000000, "training_runs": 48576}},
 "prices": {"currency": null, "per_cpu_hour": null, "per_gb_month": null,
            "compute_cost": null, "compute_limit_cost": null, "storage_cost_per_month": null},
 "caller": {"source": "token", "id": 3, "name": "agent", "used_cpu_minutes": 10,
            "budget_cpu_minutes": 10, "remaining_cpu_minutes": 0}}
```

`limit` is the account's limit this month (the plan's allowance, with overage up to its cap,
or the own budget, whichever is lower; null: unlimited), `available` what the caller can still
use with its token's or app's budget, `caller` only with an API token or app. A budget of
`null` removes it; 0 allows nothing. `ApiToken` and connections have `budget_cpu_minutes` and
`used_cpu_minutes` (this month). Only a website login can change budgets, so an agent cannot
raise its own. `overage_enabled` needs a plan with `overage_price_per_cpu_hour` and, on a
server with [paid plans](billing.md), a subscription that is paid and not cancelled.
`upgrade_url` is the pricing page on such a server (else null); the `403` of a used-up
allowance or a full storage carries it too.

The check also takes the `config` (optional) to check what depends on it, e.g. `max_samples`
against the row limit of a classical model. For classical models, only the Input layer's
`shape` is ever suggested (their output follows the data).

`POST /api/training/runs/` answers incompatible combinations with explanations, e.g.
`400 {"detail": "...", "errors": ["The network outputs 5 values, but MNIST has 10 classes. ..."],
"warnings": []}`. Starting a run while another one is active gives `409` on a server without
a training queue (see below), and `503` when training is disabled or the library the network
needs is missing (PyTorch for neural networks, scikit-learn for classical models).

**The training queue.** On a server with a training worker (`BRICKDNN_TRAINING_WORKER`, the
default with Docker Compose; see [training.md](training.md#the-training-queue)) a start does
not answer `409` because the server is busy: the run is created with the status `queued` and
trains when a worker has a free slot, runs of plans with a higher priority first. `409` then
only means that the account has as many runs waiting as it may. `GET /api/training/` has

```json
"queue": {"online": true, "workers": 1, "slots": 1, "running": 1, "waiting": 2,
          "message": null, "priority_hint": "Paid plans start first."}
```

(`null` without a worker mode: the web server trains itself). `online` false, with a
`message`, means that no worker is running: queued runs wait until one is back. `slots` are
the trainings the workers run at the same time (also `max_concurrent_runs`),
`priority_hint` (logged in) one line on what the caller's plan means for its place, null if
the plans do not differ. A queued run, in the list and alone, has

```json
"status": "queued",
"queue": {"position": 3, "waiting": 5, "estimated_wait_seconds": 240,
          "estimated_start_at": "2026-11-10T10:04:00Z", "worker_online": true,
          "blocked_by": null, "priority": 0, "hint": "Paid plans start first."}
```

`position` counts from 1 among all waiting runs. `estimated_wait_seconds` and
`estimated_start_at` are a **rough estimate** from the CPU-time estimates of the trainings
ahead and the worker's slots (null if it cannot be told, e.g. without a worker).
`blocked_by: "account"` means that the run waits for a running training of the same account
(the plan's trainings at a time). `queue` is null for every other status, and `state` is null
until the run has started. `POST …/stop/` cancels a queued run: it ends as `stopped` with the
message "Cancelled before it started." and `usage.cpu_minutes` 0. A queued run whose budget
is gone when its turn comes ends as `failed` with "Not started: …" and
`usage.stop_reason: "budget"`. A run that is started again after its worker was restarted
is `queued` again in between (its `message` says why). The training check answers with the
same `queue` object for a run that would be started now (plus `running` and `message`).

**Classical models** (Gaussian processes, SVMs, gradient boosting, see
[training.md](training.md#classical-models))
train with scikit-learn and do not need PyTorch. `GET /api/training/` describes them separately,
because `available` and `reason` are about PyTorch:

```json
"classical": {
  "available": true, "reason": null, "sklearn_version": "1.9.1",
  "max_rows": { "GaussianProcessRegressor": 5000, "GaussianProcessClassifier": 2000, "SVR": 50000, "SVC": 50000,
                "HistGradientBoostingRegressor": 1000000, "HistGradientBoostingClassifier": 1000000 },
  "tuning": false
}
```

`max_rows` gives, per classical model type, the most training rows it is fitted on (larger
training sets are subsampled); an ensemble (`VotingClassifier`, `StackingRegressor`, …) has
the strictest limit of its models, so it is not listed. The training check's
`plan.classical` lists an ensemble's `members` (with the kernel of each Gaussian process)
and, for `VotingClassifier`, `voting`. Only `val_split`, `test_split`, `max_samples` (at most
`max_rows`) and `seed` of `config` apply;
a `tuning` gives `400`. Their runs have `family` `"classical"` (otherwise `"neural"`) and
`epochs` 1.

A run's `status` is `starting`, `preparing` (loading or downloading data), `running`,
`completed`, `stopped` or `failed`. `GET /api/training/runs/<id>/` includes `state` with the
per-epoch `history` (`train_loss`, `val_loss`, `train_accuracy`, `val_accuracy`, `val_rmse`),
the current `progress`, the `best` epoch, `weights_epoch` (the epoch whose weights are tested
and saved, set when the epochs are over), `stopped_early`, the `test` result, `preprocessing`
and the last log lines. Poll it about once per second while a run is active. `state.family`
is `neural` or `classical`; a classical run is fitted once: it has no `progress`, a single
`history` entry when the fit is done, and
`state.classical` holds `estimator`, `label`, `max_rows`, `rows` (fitted on), `subsampled_from`,
`fit_seconds`, `initial_kernel`, `kernel` (fitted), `log_marginal_likelihood`,
`support_vectors`, `iterations` and `max_iterations` (gradient boosting) and `train` (the
metrics on the training rows); an ensemble adds `ensemble` (`voting` or `stacking`) and
`members` (`[{"node", "name", "type", "estimator", "label", "metrics"}]`, `metrics` being
each model's own `loss`, `accuracy`, `rmse` and `r2` on the validation rows, or on the
training rows without them), see
[training.md](training.md#ensembles-of-classical-models-in-the-graph).

Every run also has `metrics` (the last epoch while it runs, then the kept epoch, plus
`test_loss`, `test_accuracy`, `test_rmse` and `test_r2`; RMSE and R² for regressions, `null`
otherwise), `family`, `tuning` (the normalised search settings or
`null`), `trials_done` and `best_trial` (`{trial, params, val_loss, val_accuracy, val_rmse,
val_r2, weights_epoch}`). Regressions of neural networks and classical models add `val_r2` to
the `history` entries and trials and `r2` to `test`: R², the coefficient of determination,
measured against the mean of the validation or test rows themselves, `null` when all their
targets are equal (see [training.md](training.md#regression-metrics)).
In a search, `state.trials` lists every trial with its `params`, `status`,
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

A **binary classification** predicts with the run's `decision_threshold`: the second class
when its probability is at least the threshold, else the first. Every prediction adds
`positive_probability` (the probability of the second class; `probability` stays that of the
predicted class, `top` stays ordered by probability), and the answer adds `decision_threshold`,
`positive_class` and a `threshold_note`:

```json
{ "class": "fraud", "index": 1, "probability": 0.18, "positive_probability": 0.18,
  "top": [{ "class": "ok", "index": 0, "probability": 0.82 }, { "class": "fraud", "index": 1, "probability": 0.18 }] }
```

Image networks also take image files instead of arrays: `{"image": "<base64>"}` or a list of
at most 8 (PNG, JPEG, BMP, GIF or WebP, as base64 text or a `data:` URL, together at most
900 kB of text). They are converted to the channels of the network and resized to its Input
like the training images, so `input_scale` is not needed.

A **segmentation** answers with one class map per image instead of one class:

```json
{ "image": "iVBORw0KGgo…", "mask": "png" }
```

```json
{
  "run": { "id": "…", "name": "U-Net" },
  "task": "segmentation",
  "class_names": ["background", "square", "circle", "triangle"],
  "colors": ["#1a1a19", "#3987e5", "#d95926", "#199e70"],
  "predictions": [
    { "width": 64, "height": 64,
      "classes": [{ "class": "background", "index": 0, "share": 0.91 }, { "class": "circle", "index": 2, "share": 0.09 }],
      "mask_png": "data:image/png;base64,iVBORw0KGgo…" }
  ]
}
```

`classes` are the classes that occur with the share of their pixels, largest first.
`mask_png` is the class map at the size of the Input as a palette PNG: the pixel values are
the class indices, the palette shows them in `colors`. `mask` chooses how it is returned:
`png` (default), `rle` (`mask_rle`: run-lengths row by row, `[class, pixels, class, pixels,
…]`) or `none` (only the shares).

Classical models take the same rows. Classifiers give their own class probabilities (an SVC
calibrated with Platt scaling, a Gaussian process its predictive probabilities); a Gaussian
process regressor adds `"std"`, its predictive standard deviation in the units of the target,
to every prediction (`{"value": 101.3, "std": 0.4}`), and the answer a `std_note`.

**Raw rows.** A run trained on a dataset exported from a data preparation also takes rows in
the columns of the imported source table, before the steps of the preparation, with
`"input_format": "raw"` (default `"prepared"`: the feature columns of the dataset). The server
prepares them with the steps and the values they learned from the training rows, stored at the
export, then predicts (see [training.md](training.md#raw-rows-for-datasets-from-the-data-tab)):

```json
{ "input_format": "raw", "inputs": [{ "date": "2024-12-24", "store": "Köln", "promo": 1, "temp": null }] }
```

The answer adds `warnings` (unknown categories, columns not given, rows that a filter of the
preparation would have removed from the training data …). Rows that cannot be prepared are
refused with `400` and `errors`, one message per row and column:
`{"detail", "inputs", "errors": ["Row 2, 'temp': 'warm' is not a number.", …]}`.
`input_format: "raw"` for other runs is a `400` that says so.

**The prediction form.** `GET /api/training/runs/<id>/predict-form/` (everyone who can see the
run) describes the inputs for the panel of the web app: `available` and `reason`, `kind`
(`tabular`, `image`, `demo`), `task`, `class_names`, `max_rows` (64 per request),
`budget_note` (set when a compute budget is used up), for tables `prepared` (`columns` with
the training's mean and standard deviation, a few `samples` with their actual `target`) and,
for runs from a preparation, `raw` (the source columns with `kind`, the `categories` of the
training rows, `min`/`max`, `used: false` for columns the preparation drops unread, validation
rows as `samples`, `skipped_steps`, `joined` tables); for images `image` (`channels`, `height`,
`width`).

The inputs are checked and prepared like the training data from the run's
`state.preprocessing`; the network then runs in a separate process (CPU, at most 30 seconds,
`BRICKDNN_MAX_PREDICTIONS` at a time, default 1). Answers: `400 {"detail", "inputs"}` when the
inputs do not fit (`inputs` describes what the network takes), `409` when the run has no
weights or its preprocessing is unknown (e.g. it failed early), `503` when training is
disabled, PyTorch (scikit-learn for classical models) is missing or other predictions are
running (`Retry-After: 5`), `500` when the
prediction itself fails or takes too long, `429` above the rate limit.

### Explanations

**Explanations** of a trained model (see [explain.md](explain.md)): `GET
/api/training/runs/<id>/explain/` answers `{"family", "kind": "image"|"tabular"|"demo"|null,
"unavailable", "demo", "methods": [{"id", "label", "available", "reason"}]}`. `POST` computes
one method:

```json
{ "method": "tabular", "sample": 3, "target": 1 }
{ "method": "image", "input": [[0, 0, …], …], "input_scale": "raw", "layer": "conv2" }
```

| `method`      | Needs a sample | Answer                                                                 |
| ------------- | -------------- | ---------------------------------------------------------------------- |
| `samples`     | –              | `samples`: validation samples (`index`, a thumbnail `png` or the row's `values`, `label`, `predicted`, `probability`, `correct`) |
| `image`       | yes            | `image` (PNG), `prediction`, `target`, and `saliency`, `gradcam` (with `layer` and the `layers` to choose from), `occlusion`: each a `png` overlay and a coarse `grid` (0–1), or `unavailable` |
| `tabular`     | yes            | `prediction`, `base_value` (for the mean training row), `value`, `features`: `{name, value, shapley, integrated_gradients?}`, largest first |
| `permutation` | –              | `metric` (`accuracy` or `r2`), `baseline`, `rows`, `repeats`, `features`: `{name, mean, std}` |
| `boundary`    | –              | `features`, `x`/`y` ranges, `png` (the 64×64 grid), `classes` and `colors` or `value_range`, a 16×16 `grid`, `points` (training rows), `std` for Gaussian process regression |
| `activations` | yes (demo runs: no) | `nodes`: per brick `{id, type, name, shape, stats, histogram, view}`; `view` is `maps` (a PNG of the first 16 feature maps), `matrix`, `vector` or `scalar` |
| `weights`     | –              | `filters` of the first Conv2d (a PNG) and `layers` with a `histogram` and `norm` per parameter |

The sample is `sample` (the `index` of a validation sample) or `input` (one sample like an
entry of a prediction's `inputs`, with `input_scale`). `target` is the class to explain
(default: the predicted one), `layer` the brick for Grad-CAM, `node` restricts `activations`
to one brick. The answer also has `notes`, `seconds`, `cached` and `run`. Everyone who can see
the run may ask. It runs like a prediction (same process, time limit, slots and rate limit;
the CPU time counts as a prediction); answers are cached per run, and a cached answer
(`"cached": true`) costs nothing and does not count against the rate limit. Errors as for
predictions; `409` also when the method does not work for the run (the `reason` of `GET`).

## Data preparation (login required)

The Data tab imports data files (CSV, Excel, OpenDocument, JSON, JSON Lines, Parquet, Feather,
also packed, or downloaded from a link), combines and cleans them step by step and turns the result into
a training dataset. The work is done on the server with pandas (`pip install "brickdnn[data]"`),
see `backend/brickdnn/data/`. `GET /api/data/` and `GET /api/data/steps/` are public; all JSON
shapes are documented in `frontend/src/api/types.ts` (section "Data preparation").

| Method | Path                                   | Description                                         |
| ------ | -------------------------------------- | --------------------------------------------------- |
| GET    | `/api/data/`                           | Availability and limits (`DataStatus`)              |
| GET    | `/api/data/steps/`                     | All step types with their settings (`StepCatalog`)  |
| GET    | `/api/data/sources/`                   | Your imported tables, newest first                  |
| POST   | `/api/data/sources/`                   | Import a `file` with import options (multipart), or the file at a link: `{url, name?, ...options}` (JSON) |
| GET / PATCH / DELETE | `/api/data/sources/<id>/` | One source / rename it (`{name}`) / delete it with its files |
| GET    | `/api/data/preps/`                     | Your preparations, last changed first               |
| POST   | `/api/data/preps/`                     | `{source, name?, steps?, settings?}`                |
| GET / PATCH / DELETE | `/api/data/preps/<id>/`   | One preparation / change it / delete it             |
| POST   | `/api/data/preps/<id>/preview/`        | Run the (unsaved) pipeline: reports, profile, sample, readiness |
| POST   | `/api/data/preps/<id>/export/`         | Create a training dataset from the saved preparation `{name?}` |
| GET    | `/api/data/preps/<id>/download/?format=…` | `csv` (prepared data), `recipe` (JSON) or `python` (pandas script) |
| PUT / POST | `/api/uploads/?sig=…`              | Upload with a one-time link of the MCP tool `create_upload_link` (no login): the file as the body or multipart `file` |

**Import.** `file` may be CSV / TSV / text (`.csv`, `.tsv`, `.tab`, `.psv`, `.txt`), Excel
(`.xlsx`, `.xlsm`, `.xls`), OpenDocument (`.ods`), JSON (`.json`), JSON Lines (`.jsonl`,
`.ndjson`), Parquet (`.parquet`) or Feather / Arrow (`.feather`, `.arrow`), also packed as
`.gz`, `.bz2`, `.xz` or `.zip` (with one data file); `GET /api/data/` lists the endings this
server reads in `file_types`. Besides `file`, the form may contain `name`, `delimiter`
(`auto`, `,`, `;`, a tab, `|`), `decimal` (`auto`, `.`, `,`), `encoding` (`auto`, `utf-8`,
`cp1252`, `latin-1`) — these three for text files only —, `header` (`auto`, `yes`, `no`; text
files and workbooks), `sheet` (the name of a sheet of a workbook; default: the first sheet
with values) and `missing_values` (a JSON list of cell values that mean "missing", e.g.
`["-999", "n/a"]`). Left out means "detect". The answer (`201`) is the source with its typed
`columns` (`{name, kind, missing}`), the requested `import_options` and the `report` of what was
detected: `format` (`csv`, `xlsx`, `xls`, `ods`, `json`, `jsonl`, `parquet` or `feather`),
`delimiter` and `encoding` (`null` for other formats than CSV), `decimal`, `header`, `rows`,
`columns`, `warnings`, and where they apply `compression` / `member` (packed files),
`sheet` / `sheets` (the imported sheet and all sheets of a workbook), `json_path` (the keys
leading to the records, e.g. `["data"]`) and `url`. Files larger than
`BRICKDNN_DATA_MAX_UPLOAD_MB` (default 200) are refused (`413`), files with more than
`BRICKDNN_DATA_MAX_ROWS` rows (or more than `BRICKDNN_DATA_MAX_COLUMNS` columns, if set), packed
files that unpack to more than the size limit and tables with too many cells are refused
(`400`). When a sheet cannot be imported (missing, empty, too large …), the `400` answer
lists the sheets of the workbook: `{"detail": "The workbook has no sheet “Q3”.", "sheets":
["Q1", "Q2"]}`. A source that a preparation uses as its main source cannot be deleted
(`409 {"detail": "Used by the preparation “Churn”; delete it first."}`).

```bash
curl -X POST http://localhost:8080/api/data/sources/ -H "Authorization: Bearer $TOKEN" \
  -F file=@customers.csv -F delimiter=";" -F decimal="," -F 'missing_values=["-999"]'
curl -X POST http://localhost:8080/api/data/sources/ -H "Authorization: Bearer $TOKEN" \
  -F file=@sales.xlsx -F sheet=Orders
```

**Import from a link.** Send JSON instead of a form: `{"url": "https://…", "name"?, ...the
options above}`. The server downloads the file (http/https, up to
`BRICKDNN_DATA_MAX_UPLOAD_MB`) and imports it like an upload; the file name comes from the
answer's `Content-Disposition` or the link. Share links of Google Sheets (the tab of the
link's `gid` as CSV), Google Drive, GitHub (`blob` pages), Hugging Face (`blob` pages) and
Dropbox are turned into download links. Links with a user name or password and links to
addresses that are not public (private networks, loopback, link-local such as cloud metadata,
also after a redirect) are refused (`400` with the reason), as are web pages (HTML), missing
files (`404` of the other server) and files that are too large. `403` when link imports are
switched off (`BRICKDNN_DATA_URL_IMPORT=false`, `url_import: false` in `GET /api/data/`);
`BRICKDNN_DATA_URL_ALLOW_PRIVATE=true` allows private addresses for installations in a trusted
network. Link imports count against the same rate limit as uploads. Details:
[data preparation](data-preparation.md#security-of-link-imports).

```bash
curl -X POST http://localhost:8080/api/data/sources/ -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.google.com/spreadsheets/d/<id>/edit#gid=0", "name": "Survey"}'
```

Database connections (PostgreSQL, MySQL …) are not supported yet.

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

**Upload links.** `PUT /api/uploads/?sig=…` with the file as the body (`curl -T file`) or
`POST` with a multipart field `file` (`curl -F file=@file`) needs no login: the signed link
(MCP tool `create_upload_link`, see [mcp.md](mcp.md#uploading-a-dataset)) names the user, the
purpose (`data_source` or `training_dataset`), the file name and the import options, and works
once within 15 minutes. The upload is imported like `POST /api/data/sources/` or
`POST /api/training/datasets/` (same limits, checks and rate limit) and answered with `201` and
the compact JSON of the MCP tools (`source`, or `dataset` with its `reference`). `403` for an
expired, invalid or used link or an inactive user, `413` for a file over the limit, `429` when
too many files were uploaded.

All data endpoints except `GET /api/data/` answer `503` when data preparation is switched off
(`BRICKDNN_DATA_PREP=false`) or pandas is not installed; `GET /api/data/` then says why
(`available: false`, `reason`).

## Rate limits

Anonymous clients: 300 requests/minute, logged-in users: 600/minute, login and registration:
20/minute, starting training runs: 120/hour, dataset uploads, data imports (files and links) and exports of
preparations: 60/hour, predictions: 120/hour, previews and downloads of preparations:
1200/hour, MCP requests: 600/hour, OAuth registration, token and revocation requests:
120/minute, changes to the own account (e-mail, password, export, deletion): 30/hour,
checkout and changes of a paid plan: 60/hour (`BRICKDNN_BILLING_RATE`), asking for a password
reset link: 10/hour per IP address (`BRICKDNN_PASSWORD_RESET_RATE`), sending the confirmation
link again: 5/hour (`BRICKDNN_EMAIL_VERIFY_RATE`), the second step of a two-factor login:
30/hour per IP address (`BRICKDNN_TWO_FACTOR_RATE`). Adjust
with `BRICKDNN_ANON_RATE`, `BRICKDNN_USER_RATE`, `BRICKDNN_AUTH_RATE`,
`BRICKDNN_TRAINING_RATE`, `BRICKDNN_UPLOAD_RATE`, `BRICKDNN_PREDICT_RATE`,
`BRICKDNN_DATA_RATE`, `BRICKDNN_MCP_RATE`, `BRICKDNN_OAUTH_RATE`, `BRICKDNN_ACCOUNT_RATE` and
`BRICKDNN_SHARING_RATE` (invitations and link changes, 120/hour)
(e.g. `100/minute`).
