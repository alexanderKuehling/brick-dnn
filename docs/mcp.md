# MCP server and API tokens

Brick DNN has a built-in [MCP](https://modelcontextprotocol.io) server, so AI assistants such
as Claude can build networks, check them, generate PyTorch code, save them in your account
and train them on your server. Scripts can use the same **API tokens** to call the
[REST API](api.md).

## 1. Create an API token

Log in, open the account menu → **API tokens** and create a token:

- **Full access** can also save networks and start or stop trainings. **Read only** can look
  at everything (layers, examples, your networks and training runs), analyse networks and
  generate code, but change nothing.
- Choose an expiry (7, 30, 90 days, 1 year or never). You can revoke a token at any time;
  apps using it stop working immediately.
- The token (`bdnn_…`) is shown **once**. Treat it like a password.

Only a login on the website can create, list or revoke tokens, not a token itself.

## 2. Connect an assistant

The server is at `https://<your domain>/api/mcp/` (Streamable HTTP). Send the token as
`Authorization: Bearer <token>`.

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

The server only supports API-token login, not OAuth. Clients that can only add remote
servers through an OAuth sign-in cannot connect yet.

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

Read-only tokens only see the read tools. Networks use the [graph format](graph-format.md).
Everything happens as the token's owner: an assistant sees and changes only that user's
networks, datasets and runs, with the same limits and validation as the web app.

## Details

- JSON-RPC 2.0 over HTTP `POST`, answered with `application/json`. There are no sessions and
  no event streams (`GET` answers `405`); notifications get `202`. Protocol versions
  `2025-11-25`, `2025-06-18`, `2025-03-26` and `2024-11-05`.
- Requests from browsers must come from this site or a `CORS_ALLOWED_ORIGINS` origin.
- Rate limits: 600 MCP requests per hour per user (`BRICKDNN_MCP_RATE`), plus the usual limit
  for starting trainings (`BRICKDNN_TRAINING_RATE`).
- Only a SHA-256 hash of each token is stored. Admins can see and revoke tokens in the Django
  admin but never see the token itself.
