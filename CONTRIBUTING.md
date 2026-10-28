# Contributing to Brick DNN

Thanks for your interest! Bug reports, ideas, new layers, documentation and code are all
welcome.

## Ways to help

- **Report a bug** or **request a layer** via [GitHub issues](https://github.com/alexanderKuehling/brick-dnn/issues).
  For problems with a specific network, attach its JSON file (*File → Save as JSON file*).
- **Add a layer**: the most common contribution, see [docs/adding-a-layer.md](docs/adding-a-layer.md).
- **Improve the editor**: UX ideas are listed in the README roadmap.

## Development setup

```bash
make install        # backend venv + npm install
make backend        # terminal 1: API on :8000
make frontend       # terminal 2: UI on :5173
```

Or follow the manual steps in the [README](README.md#for-development). Installing PyTorch and
torchvision in the backend venv is strongly recommended, because the tests then execute and
train every generated model (and the training panel in the UI works):

```bash
make install-training   # or, CPU-only: backend/.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Before you open a pull request

```bash
make test           # backend: pytest; frontend: typecheck, prettier, eslint, stylelint, vitest
make lint           # ruff check + ruff format --check
```

- Keep pull requests focused on one change and describe what and why.
- Add tests for new behaviour. For layers, one line in `CASES` (`backend/tests/test_layers.py`)
  already verifies the shape inference and the generated code against PyTorch.
- Format code with `ruff format` (backend) and `npm run prettier:write` (frontend).
- User-facing messages should say what is wrong *and* how to fix it.

## Project principles

- **The layer catalogue is the single source of truth.** The frontend must not hard-code layer
  parameters; it renders whatever `GET /api/layers/` returns.
- **Generated code is for humans.** It should look like code a careful person would write:
  readable names, no unused imports, PEP 8.
- **`brickdnn` stays dependency-free** so it can be used anywhere. PyTorch is only imported by
  the training engine, runner and CLI (`brickdnn/training/`), never by the web server process.

By contributing you agree that your contributions are licensed under the [Apache License 2.0](LICENSE)
and to follow the [code of conduct](CODE_OF_CONDUCT.md).
