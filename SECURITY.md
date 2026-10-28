# Security policy

If you find a security issue, please **do not open a public issue**. Report it privately via
[GitHub security advisories](https://github.com/alexanderKuehling/brick-dnn/security/advisories/new)
instead. You will get an answer as soon as possible.

## Running your own instance

- Never set `DJANGO_DEBUG=true` on a server reachable from the internet.
- Set `DJANGO_ALLOWED_HOSTS` and `DJANGO_CSRF_TRUSTED_ORIGINS` to your domain and serve the app
  over HTTPS (e.g. behind a reverse proxy). With HTTPS, also set `DJANGO_SECURE_COOKIES=true`.
- If you put another reverse proxy in front of the bundled nginx, increase
  `BRICKDNN_NUM_PROXIES` so that rate limits see the real client address.
- Keep the data volume private: it contains the database and the generated secret key.
- Disable public sign-up with `BRICKDNN_ALLOW_REGISTRATION=false` if you do not need it.
- Code generation only produces text. **Training** (enabled when PyTorch is installed) executes
  the generated model code in a separate process. That code is built from validated layer
  parameters only (numbers, choices and identifiers), never from Python supplied by users.
- With training enabled, every logged-in user can use the server's CPU/GPU and upload datasets
  (`BRICKDNN_MAX_UPLOAD_MB`). On a public server, disable registration or set
  `BRICKDNN_TRAINING=false`. Uploaded ZIP files are extracted with checks against path
  traversal, symbolic links and oversized contents.
