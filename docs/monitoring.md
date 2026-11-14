# Error monitoring with Sentry

A server can report its errors to [Sentry](https://sentry.io): unhandled exceptions of the web
backend, the realtime server, the training worker and the daily clean-up, and errors in the
browser app, each with the request, release and code line that caused it, plus a sample of
traces (how long page loads and API requests took) and the server's log messages.

**It is off unless a DSN is set.** Without one nothing is loaded in the browser and nothing is
sent; local builds never send anything.

## What is sent, and what is not

- **Sent:** the error with its stack trace, the URL and route of the request, the release and
  environment, browser and operating system (from the user agent), the country (Sentry derives
  it from the IP address before dropping the address), timings of sampled traces,
  and log messages of the server (`logging`, INFO and up).
- **Not sent:** who it was. No user id, name or e-mail address (`send_default_pii=False` in the
  backend; `dataCollection` with `userInfo: false` in the browser), no IP address (also dropped
  by both Sentry projects: *Prevent storing of IP addresses*), no cookies, no request headers or
  query strings from the browser, and no request bodies (form data, JSON, uploads:
  `max_request_body_size="never"`). Sentry's scrubbers remove values that look like
  passwords, tokens or keys from what is left.
- **No Session Replay** and no user feedback widget: they would record what people do.
- **Health checks** (`/api/health/`) are not traced.

The privacy policy has to name Sentry as a processor (for brickdnn.com: section 4 of
`deploy/sites/brickdnn.com/legal/privacy.md`). Sentry's data processing addendum is part of
its terms; accept or download it in Sentry under *Settings → Legal & Compliance*. Choose the
EU data region when you create the organization (`de.sentry.io`), as for brickdnn.com.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `BRICKDNN_SENTRY_DSN` | – (off) | DSN of the Sentry project for the server (Python, Django) |
| `BRICKDNN_SENTRY_FRONTEND_DSN` | – (off) | DSN of the project for the browser app (React); built into the app when the frontend image is built |
| `BRICKDNN_SENTRY_ENVIRONMENT` | `production` (`development` with `DJANGO_DEBUG`) | Environment the events are filed under |
| `BRICKDNN_SENTRY_TRACES_SAMPLE_RATE` | `0.1` (`1.0` in development) | Share of requests and page loads that are traced (0 to 1) |
| `BRICKDNN_SENTRY_RELEASE` | – | Release of the events; every deploy sets it to `brick-dnn@<commit>` |
| `SENTRY_AUTH_TOKEN` | – (no upload) | **Secret.** Lets the frontend build upload its source maps ([below](#readable-stack-traces-in-the-browser)); only used while the image is built |

A DSN is not a secret (the browser app contains it anyway): it only allows sending events to
the project. With the *Deploy server* workflow, set them as repository **variables**
(`BRICKDNN_SENTRY_DSN`, `BRICKDNN_SENTRY_FRONTEND_DSN`, optionally the environment and the
rate): every deploy writes them into the server's `.env`
([Settings from GitHub](../deploy/server/README.md#settings-from-github)). By hand, put them
into `.env` and run `docker compose up -d --build` (the frontend has to be built again for its
DSN).

The browser app sends trace headers (`sentry-trace`, `baggage`) only with requests to its own
API (`/api/…`), so a page load and the API requests it makes appear as one trace.

## Checking that it works

After a deploy with the DSNs set, an error has to show up in Sentry's *Issues* within a
minute. A quick check for the server (a management command loads the same settings):

```bash
docker compose exec backend python manage.py shell -c \
  "import sentry_sdk; sentry_sdk.capture_message('Sentry check from the server'); sentry_sdk.flush()"
```

For the browser app, open the site, then in the browser's console:
`setTimeout(() => { throw new Error('Sentry check from the browser') })`.

## Readable stack traces in the browser

The production build of the app is minified. So that Sentry shows browser errors with the
original file, line and code, the image build uploads the app's source maps to Sentry
(`@sentry/vite-plugin`, `frontend/vite.config.mjs`), to the organization `brickdnn` in the EU
region (`https://de.sentry.io`), project `brickdnn-frontend`, under the same release as the
events (`brick-dnn@<commit>`).

- **Only with a token.** The upload needs `SENTRY_AUTH_TOKEN` and a release
  (`BRICKDNN_SENTRY_RELEASE`, written by every deploy) while the frontend image is built.
  Without them the build makes no source maps and talks to nobody: local builds, CI and
  servers without the token need nothing.
- **Never served.** The maps are *hidden* (the bundles have no `sourceMappingURL` comment) and
  are taken out of `dist` after the upload, so nginx has none to hand out.
- **A failed upload does not stop the deploy:** the build warns
  (`Sentry: source maps not uploaded: …`) and the site goes up with stack traces as before.
- **The token stays out of the image.** Docker Compose passes it as the build secret
  `sentry_auth_token` (`docker-compose.yml`, from the variable `SENTRY_AUTH_TOKEN`), which
  `frontend/Dockerfile` mounts for the build command only: it is not a build argument and is in
  no layer or image history.

### Creating the token

The token only needs to create releases and upload files to them:

1. In Sentry, *Settings → Developer Settings → Organization Tokens* (of `brickdnn`), *Create
   New Token*, named after the server (e.g. `brickdnn.com deploy`). Organization tokens have
   one fixed scope, `org:ci`, made for exactly this: creating releases and uploading source
   maps. They cannot read events or change settings.
2. Copy it (`sntrys_…`); Sentry shows it once.

A personal token works too (*User settings → Personal Tokens*), with the scope
`project:releases` and nothing else, but it belongs to your account and ends with it, so the
organization token is the better choice.

### Where to put it

- **With the *Deploy server* workflow:** as the repository **secret** `SENTRY_AUTH_TOKEN`
  (*Settings → Secrets and variables → Actions*, tab *Secrets*). Every deploy writes it into
  the server's `.env` ([Settings from GitHub](../deploy/server/README.md#settings-from-github)),
  so the next deploy already uploads. The deploy script on the server has to be the current
  one: an older copy does not take this name, and the workflow then leaves it out with a
  warning instead of sending it (the deploy goes on, without the upload).
- **By hand:** in the server's `.env` (`SENTRY_AUTH_TOKEN=sntrys_…`), then
  `docker compose -f docker-compose.yml -f deploy/server/compose.yml up -d --build`.

To check: after a deploy, the project's source maps in Sentry (*Settings → Projects →
brickdnn-frontend → Source Maps*) list a bundle for the new release, and the next browser
error shows the original code. If not, the build log says why
(`docker compose -f docker-compose.yml -f deploy/server/compose.yml build --progress=plain frontend`,
look for `Sentry:`). Server errors are readable as they are.

To revoke the token, delete it in Sentry and remove it from GitHub and `.env`: builds then go
on without uploading.
