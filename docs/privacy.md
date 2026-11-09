# Privacy and legal pages

For operators of a Brick DNN server that other people use: what the app stores about them,
where and for how long, what users can do themselves, how to publish an imprint, privacy
policy and terms of use, and what is left for you to do. The technical parts are built in;
the legal texts are yours to write. German templates and a checklist for the privacy policy
are in [`deploy/legal/`](../deploy/legal/README.md). None of this is legal advice.

## What is stored, where, for how long

| Data | Where | Kept until |
| ---- | ----- | ---------- |
| **Accounts:** username, e-mail address, password hash, sign-up, last login and last activity, accepted terms of use (time and version) | database | the account is deleted: by its owner, an admin, or the purge of inactive accounts |
| **Logins:** refresh tokens (with their id, user and expiry) | database | 7 days after their last use; `purge_old_data` removes expired ones daily. Access tokens (30 minutes) are not stored |
| **Networks:** graph, description, version history, comments, members, share link | database | deleted by their owner (or with the account) |
| **Data tab imports:** the uploaded or downloaded file (CSV, Excel, JSON, Parquet ...), the link it came from (without its query) and the table read from it | `BRICKDNN_DATA_DIR/prepare/sources/<id>/` | deleted by the user, with the account, or when unused (below) |
| **Data preparations:** steps and settings | database | deleted by the user, with the account or with their source |
| **Training datasets:** an uploaded CSV file or the images of a ZIP file | `BRICKDNN_DATA_DIR/training/datasets/<id>/` | as imports |
| **Training runs:** network, settings and metrics; job, state, log and trained weights | database; `BRICKDNN_DATA_DIR/training/runs/<id>/` | as imports (finished runs only) |
| **API tokens and connected apps:** name, start of the token, SHA-256 hash, dates, the app's name, their monthly budget | database | revoked by the user, or with the account |
| **Agent activity log:** every MCP tool call and API request made with a token or app: token or app name, start of the token, user agent, tool or route, outcome and error message, duration, ids and names of what it touched, summarised arguments (sizes and counts, never file contents, prediction inputs or secrets) | database | `BRICKDNN_AGENT_LOG_DAYS` (90) days, at most 20,000 entries per account; with the account. Kept when the token is revoked |
| **Usage records:** the CPU time of each training run and prediction per calendar month, the part beyond the plan's allowance, which token or app caused it (with its name), the run's name; the account's plan, budgets and limits | database | with the account (they stay when a run, token or app is deleted: the month's total and later billing need them) |
| **Paid plans** (only with [billing](billing.md)): the account's customer and subscription ids at Paddle, plan, status and dates; for every notification from Paddle its type, time and subscription id (not its content); overage charges (month, CPU hours, amount) | database | the account is deleted. Paddle keeps payment data and invoices itself |
| **Rate-limit counters:** IP address (anonymous requests) or user id | file cache in `BRICKDNN_DATA_DIR/cache` | at most an hour |
| **Live editing:** who is in a network, what they select or drag | Redis, in memory only | 30 seconds; never written to disk |
| **Access log** (nginx): shortened IP address, time, path without query string, status, size, duration, user agent | Docker's container logs | at most 3 × 10 MB per container |

The browser keeps the login (`brickdnn:tokens`), the builder's draft (`brickdnn:draft`), the
last training settings and data preparation and the colour scheme in `localStorage`. The app
sets no cookies (the Django admin at `/admin/` does, for the operator), loads nothing from
other servers, has no analytics and sends no e-mails. AI assistants a user connects (MCP,
OAuth or an API token) fetch data on the user's behalf; their provider is responsible for it.
The one exception to "nothing from other servers" is Paddle's checkout on a server that sells
plans, see [Paid plans](#paid-plans-paddle).

The [online demo](../deploy/huggingface/README.md) is a static page without accounts: it
stores nothing on a server, keeps the draft in the browser and loads Pyodide from
cdn.jsdelivr.net. It shows no legal links; Hugging Face's terms apply to the Space.

## Legal pages

Put Markdown files into `BRICKDNN_LEGAL_DIR`:

| File | Page | Notes |
| ---- | ---- | ----- |
| `imprint.md` | `/imprint` | Impressum (§ 5 DDG in Germany) |
| `privacy.md` | `/privacy` | privacy policy |
| `terms.md`   | `/terms`   | terms of use; sign-up then asks to accept them |

| Setup | `BRICKDNN_LEGAL_DIR` | How to put the files there |
| ----- | -------------------- | -------------------------- |
| Server (`deploy/server/`) | `/srv/brick-dnn/legal` on the server, then `deploy/sites/<domain>/legal` of the deployed checkout, both read-only | commit them to `deploy/sites/<domain>/legal/` (online with the next deploy), or `scp *.md root@SERVER:/srv/brick-dnn/legal/` (online at once, takes precedence) |
| Docker Compose | `/data/legal` in the data volume | `docker compose exec backend mkdir -p /data/legal`, then `docker compose cp privacy.md backend:/data/legal/` |
| Without Docker | `backend/data/legal` (`legal/` in `BRICKDNN_DATA_DIR`) | copy them there |

The files are read on every request: changes are online at once. A missing or empty file
(or one over 512 KB) means the page is not offered: the app links only the pages that exist
(header menu, phone menu, login, sign-up and account pages). The public API is
`GET /api/legal/` and `GET /api/legal/<page>/` ([api.md](api.md#legal-pages-public)).

The app renders a safe subset of Markdown into React elements: headings, paragraphs (a single
line break stays a line break, for addresses: write a paragraph on one line, as the templates
do), lists, quotes, rules, bold, italic, code, links
to `https://`, `mailto:`, `tel:` and paths like `/privacy`, and plain web and e-mail
addresses. HTML is shown as text, never run; other link targets (e.g. `javascript:`) become
plain text. There are no tables or images.

**Terms of use.** With `terms.md`, sign-up shows a required checkbox ("I agree to the terms
of use and have read the privacy policy"), and the API requires `accept_terms: true`. The
account's profile stores when, and the versions of `terms.md` and `privacy.md` (the start of
the SHA-256 hash of each file: every edit is a new version). Existing accounts are not asked
again when the terms change. Without `terms.md` there is no checkbox (a local installation
stays as simple as before).

## What users can do themselves

The account page (`/account`, in the account menu) offers, after a website login (API tokens
cannot), rate limited with `BRICKDNN_ACCOUNT_RATE` (30/hour):

- **Change the e-mail address**, e.g. for accounts from before e-mail addresses were
  required. It must be unused, like at sign-up.
- **Change the password** (with the current one). All other logins end (their refresh tokens
  are blacklisted; their access tokens expire within 30 minutes); this one goes on.
- **Download my data:** a ZIP file with `README.txt` and JSON files for the account, own
  networks with their history, memberships, own comments, API tokens (no secrets), connected
  apps, the agent activity log, data sources, preparations, datasets, training runs
  (settings, metrics; not the weights), the usage records with the plan and budgets
  (`usage.json`) and, for accounts that bought a plan, `billing.json` (below), plus the
  uploaded files themselves up to `BRICKDNN_EXPORT_MAX_FILES_MB` (200 MB)
  in total. Files beyond that are listed in `files.json` with a note. Nothing of other users
  is included (e.g. their comments on the user's networks).
- **Delete the account**, with the password and the username typed again. Running trainings
  are stopped, then everything the account owns is deleted, files included: its networks
  (members lose access, people editing them live see that the network is gone), data
  sources, preparations, datasets, training runs, API tokens, connected apps, the agent
  activity log, usage records, memberships and its comments. Comments are deleted, not anonymised: a comment's text can itself be
  personal data. Replies of others to a thread the user started go with the thread, as when
  the user deletes the thread. Networks of others the user worked on stay; their history
  keeps the versions without saying who saved them. All refresh tokens are blacklisted.
  A subscription to a paid plan is cancelled at Paddle first, at once; if Paddle cannot be
  reached, nothing is deleted and the user is asked to try again.

## Automatic deletion (optional)

Off by default. In `.env` (Docker Compose passes them on):

| Variable | Deletes |
| -------- | ------- |
| `BRICKDNN_DELETE_UNUSED_UPLOADS_DAYS` | data sources (with the preparations built on them), datasets and finished training runs nobody has used for this many days |
| `BRICKDNN_DELETE_INACTIVE_ACCOUNTS_DAYS` | accounts nobody has logged in to or used for this many days, with everything they own (as above) |

"Used" means: a data source was opened or its table read (preview, export, join) or a
preparation using it was opened or changed; a dataset was opened or trained on; a run was
opened, predicted with or its weights downloaded, or it finished. An account is used when
someone logs in, a login is refreshed (the app does so at least every half hour while it is
used), or one
of its API tokens or connected apps is used. Staff accounts are never deleted automatically,
and nothing an active training uses is. Times are tracked from the update that added them:
existing data and accounts count as used at the time of the migration. No one is warned by
e-mail (the app sends none): name the periods in your privacy policy and terms of use; the
upload dialogs and the account page mention them, too.

Docker Compose runs a small `cleanup` service (the backend image, sleeping in between) that
calls `python manage.py purge_old_data` ten minutes after it starts and then once a day. It
logs every deletion with ids, never names: `docker compose logs cleanup`. By hand:

```bash
python manage.py purge_old_data --dry-run                 # what would be deleted
python manage.py purge_old_data --unused-uploads-days 90  # override the setting once
```

Expired logins (refresh tokens, unused OAuth codes) and expired admin sessions are cleaned up
on every run, whatever the settings. So are entries of the **agent activity log** older than
`BRICKDNN_AGENT_LOG_DAYS` (default 90; it is on by default, as users see it on the page *Agent
activity*; `0` turns the log off and the next run deletes it) and the oldest entries of an
account beyond 20,000. `--agent-log-days N` overrides the setting once.

## Logs and security headers

- **Log size:** Docker keeps at most 3 × 10 MB of logs per container (`logging` in both
  compose files), so IP addresses in logs do not pile up.
- **nginx** logs IP addresses shortened (IPv4: the last block is 0; IPv6: only the first two
  blocks), behind Caddy also the client address from `X-Forwarded-For`, the path without
  query string (signed download links stay out of the log) and no referrer.
- **Caddy** writes no access log; its error log (e.g. failed TLS handshakes) contains full IP
  addresses and is limited like the others.
- gunicorn and Daphne only see nginx's internal address.
- The app's pages and assets are sent with `Referrer-Policy: strict-origin-when-cross-origin`,
  `X-Content-Type-Options: nosniff` and `Permissions-Policy: camera=(), microphone=(),
  geolocation=(), payment=(…), usb=()`, next to `Content-Security-Policy: frame-ancestors
  'none'` (`frontend/security-headers.conf`). The payment sheet of the browser (Apple Pay,
  Google Pay) is allowed for the frames of Paddle's checkout only, not for the app. Caddy
  adds HSTS.

The server itself logs, too (SSH logins with IP addresses in the systemd journal); limit the
journal if you like (`SystemMaxUse=` in `/etc/systemd/journald.conf`).

## Paid plans (Paddle)

Only on a server that sells plans ([billing.md](billing.md)); without the Paddle settings none
of this happens. Paddle (Paddle.com Market Ltd, United Kingdom) is the seller of the plans, the
*merchant of record*: the contract of sale, the payment, the invoice and the tax are between
the buyer and Paddle, which is a controller of its own for that, not a processor of the
operator.

- **What the server sends to Paddle:** when a user opens the checkout for the first time, the
  account's e-mail address and its numeric id, to create a customer there. A changed e-mail
  address is passed on. Nothing else about the account goes there.
- **What the user gives Paddle directly:** payment data, country and postcode or billing
  address, a tax number if any. None of it reaches this server; the operator sees the
  purchase (e-mail address, country, amounts, status) in the Paddle dashboard.
- **Third-party content:** Paddle.js (`cdn.paddle.com`) and the checkout frame are loaded
  only when a user opens the checkout, or follows a payment link from Paddle to `/pricing`.
  Looking at the pricing page or the account page loads nothing from Paddle. Paddle then
  sees the IP address and browser and may set cookies of its own. The customer portal
  ("Payment method and invoices") is a page at Paddle, opened in a new tab.
- **What the server stores:** see the table above. `billing.json` in the data export holds
  it; deleting the account deletes it and cancels the subscription. Paddle keeps its own
  records (invoices, payments) for as long as it must.
- **E-mails:** receipts, invoices and payment reminders are sent by Paddle, not by the app.
- **Notifications:** Paddle calls `/api/billing/paddle/webhook/`; the access log shows these
  requests like any other.

The privacy policy has to say this (the template has a section), and the terms of use need a
part on paid plans; see the checklist in [billing.md](billing.md#privacy-and-legal-texts).

## Left for the operator

- Write the imprint, privacy policy and terms of use (generator or lawyer; templates and
  checklist in [`deploy/legal/`](../deploy/legal/README.md)) and put them on the server.
- Sign the data processing agreements (AV-Vertrag, Art. 28 GDPR) with Hetzner (in the Hetzner
  console) and, if you use it, Supabase (their DPA); check where Supabase keeps the data and
  on what basis it may leave the EU.
- Decide on the deletion periods (or none) and state them in the texts. Mention backups
  (Hetzner backups keep seven, Supabase its own) and how long they keep deleted data.
- Keep a short record of processing activities (Art. 30) and of the technical and
  organisational measures; this page is a starting point.
- Have a plan for data breaches: report to the supervisory authority within 72 hours
  (Art. 33), inform the people affected if the risk is high (Art. 34).
- Answer requests of data subjects within a month; access, portability, correction of the
  e-mail address and erasure are self-service on `/account`. Usernames cannot be changed by
  users: rename an account in the Django admin if someone asks.
