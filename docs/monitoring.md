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

The production build of the app is minified, so browser errors show minified frames until
source maps are uploaded to Sentry at build time. That needs a Sentry auth token on the
server during the build; it is not set up yet (see the Sentry docs on
[source maps for Vite](https://docs.sentry.io/platforms/javascript/guides/react/sourcemaps/)).
Server errors are readable as they are.
