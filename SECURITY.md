# Security policy

If you find a security issue, please **do not open a public issue**. Report it privately via
[GitHub security advisories](https://github.com/alexanderKuehling/brick-dnn/security/advisories/new)
instead. You will get an answer as soon as possible.

## Running your own instance

- Never set `DJANGO_DEBUG=true` on a server reachable from the internet.
- Set `DJANGO_ALLOWED_HOSTS` and `DJANGO_CSRF_TRUSTED_ORIGINS` to your domain and serve the app
  over HTTPS (e.g. behind a reverse proxy). With HTTPS, also set `DJANGO_SECURE_COOKIES=true`.
- Keep the data volume private: it contains the database and the generated secret key.
- Disable public sign-up with `BRICKDNN_ALLOW_REGISTRATION=false` if you do not need it.
- The server never executes the generated code; it only produces text.
