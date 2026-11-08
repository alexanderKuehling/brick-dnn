# MCP server, OAuth and API tokens

Brick DNN has a built-in [MCP](https://modelcontextprotocol.io) server, so AI assistants such
as Claude can build networks, check them, generate PyTorch code, save them in your account
and train them on your server. The server is at `https://<your domain>/api/mcp/`
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
change things (save networks, start and stop trainings) or only look, and allow it.

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

- **Full access** can also save networks and start or stop trainings. **Read only** can look
  at everything (layers, examples, your networks and training runs), analyse networks and
  generate code, but change nothing.
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

| Tool                    | Access     | What it does                                                         |
| ----------------------- | ---------- | -------------------------------------------------------------------- |
| `list_layer_types`      | read       | All layer types with their parameters                                |
| `list_examples`         | read       | Example networks with their graphs                                   |
| `analyze_network`       | read       | Validate a graph, infer shapes and parameter counts, explain errors  |
| `generate_pytorch_code` | read       | PyTorch script for a valid graph                                     |
| `list_projects`         | read       | Your saved networks                                                  |
| `get_project`           | read       | One saved network with its graph                                     |
| `save_project`          | full       | Create a network or update one (appears in the web app)              |
| `get_training_options`  | read       | Whether training is available, datasets, default settings, search    |
| `check_training`        | read       | Does a network fit a dataset?                                        |
| `start_training`        | full       | Train in the background                                              |
| `list_training_runs`    | read       | Your training runs with status and metrics                           |
| `get_training_run`      | read       | One run with its metrics per epoch                                   |
| `stop_training`         | full       | Stop a run after the current batch                                   |

Read-only tokens and apps allowed read-only access only see the read tools. Networks use the [graph format](graph-format.md).
Everything happens as the token's owner: an assistant sees and changes only that user's
networks, datasets and runs, with the same limits and validation as the web app.

## Details

- JSON-RPC 2.0 over HTTP `POST`, answered with `application/json`. There are no sessions and
  no event streams (`GET` answers `405`); notifications get `202`. Protocol versions
  `2025-11-25`, `2025-06-18`, `2025-03-26` and `2024-11-05`.
- Requests from browsers must come from this site or a `CORS_ALLOWED_ORIGINS` origin.
- Rate limits: 600 MCP requests per hour per user (`BRICKDNN_MCP_RATE`), plus the usual limit
  for starting trainings (`BRICKDNN_TRAINING_RATE`); app registration, token and revocation
  requests: 120 per minute per IP address (`BRICKDNN_OAUTH_RATE`).
- Only SHA-256 hashes of tokens, refresh tokens, codes and client secrets are stored. Admins
  can see and revoke tokens, apps and connections in the Django admin but never see a token.
- The web app may not be shown in frames of other sites (`frame-ancestors 'none'`), so the
  consent screen cannot be overlaid.
