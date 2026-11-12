# E-mail, password reset and two-factor login

What the server sends by e-mail, how to set up sending with an e-mail provider (with the DNS
records a domain like brickdnn.com needs), and how accounts are protected with two-factor
authentication. Without e-mail settings the server sends nothing and everything works as
before, except what needs e-mail (it says so).

## What is sent

| E-mail | To | When |
| --- | --- | --- |
| **Confirm your e-mail address** | the new address | sign-up, a changed address, "Send me a new link" on the account page (at most 5 an hour) |
| **Reset your password** | the account's address, if it may get one (below) | "Forgot password?" on the login page; at most 3 an hour per account |
| **Your password was changed** | the account's address | the password was changed on the account page |
| **Your e-mail address was changed** | the **old** address | the address was changed on the account page |
| **Two-factor authentication turned on / off** | the account's address | on the account page, or `manage.py reset_two_factor` |
| **A new API token was created / An app was connected** | the account's address | a token was created, an app (e.g. Claude) was connected with OAuth |
| **Your account will be deleted** | the account's address | `BRICKDNN_INACTIVE_ACCOUNT_WARNING_DAYS` (14 and 3) days before an inactive account is deleted (`BRICKDNN_DELETE_INACTIVE_ACCOUNTS_DAYS`) |
| **Server errors** | `BRICKDNN_ADMIN_EMAIL` (you) | an unhandled error (HTTP 500), at most one e-mail per kind of error an hour |

All of them are short, in English like the app, plain text with a simple HTML version, with
the site name (`BRICKDNN_SITE_NAME`, default "Brick DNN") and links to `BRICKDNN_PUBLIC_URL`.
Receipts and invoices of paid plans come from Paddle, not from the app.

**Which address may receive reset links.** A link to choose a new password goes to an address
that was **confirmed** (its owner opened the link we sent to it while logged in), or to the
unchanged address of an account that existed before this feature (those could never confirm
theirs). A reset link to an address someone typed and never confirmed would hand the account
to whoever owns that address (a typo is enough), so new accounts must confirm theirs first;
the account page and the "Forgot password?" page say so. Opening a reset link counts as
confirming the address. Security notices follow the same rule; deletion warnings go to any
address of the account (they say nothing secret and keep accounts from being lost).
Everything else (logging in, invitations by e-mail, the checkout of paid plans) works with
unconfirmed addresses as before.

**Confirming needs the account.** The link in "Confirm your e-mail address" confirms the
address only for the logged-in account it was sent to: whoever opens it is asked to log in.
A stranger who got the link because of a typo cannot confirm it for someone else's account.

**Sending.** The server hands e-mails to the SMTP server in a background thread, so no page
waits for it (and "Forgot password?" answers as fast for addresses that do not exist). A
failure is logged (`Could not send the e-mail …`, without the address) and never shown as
an error. The daily clean-up sends its warnings directly and only counts an account as
warned when the e-mail went out. E-mails are not stored on the server.

**Deleting inactive accounts with e-mail.** With e-mail set up, an account with an address is
only deleted once a warning went out at least 3 days before (the smallest of
`BRICKDNN_INACTIVE_ACCOUNT_WARNING_DAYS`) and after its last use: nobody is deleted without
notice. Accounts that are overdue when you switch e-mail on are warned first and deleted 3
days later. Logging in keeps the account and starts over. If sending fails, the account is
kept. Accounts without an address are deleted as before. `docker compose logs cleanup` shows
`Warned account #12 …` lines.

## Setting up sending

Any provider with SMTP works. For a site in the EU an EU provider makes the privacy policy
simpler (data processing agreement, no transfer to a third country):

| Provider | Server, port | Login | Password | Notes |
| --- | --- | --- | --- | --- |
| [Brevo](https://www.brevo.com) (France) | `smtp-relay.brevo.com`, `587` | the SMTP login shown under *SMTP & API* | an SMTP key created there | free plan with 300 e-mails a day |
| [Mailjet](https://www.mailjet.com) (France) | `in-v3.mailjet.com`, `587` | API key | secret key | free plan with 200 e-mails a day |
| [Postmark](https://postmarkapp.com) (USA) | `smtp.postmarkapp.com`, `587` | server API token | the same token | very good delivery, no free plan beyond a test |
| [Amazon SES](https://aws.amazon.com/ses/) | `email-smtp.eu-central-1.amazonaws.com`, `587` | SMTP user name | SMTP password (not the AWS keys) | starts in a sandbox: request production access |

Use the values your provider's dashboard shows; they change now and then.

### 1. Prove that you own the domain (DNS)

At the provider, add the domain (e.g. `brickdnn.com`) as a sender domain. The dashboard then
lists DNS records; add them where the DNS of the domain is managed (where the `A` record of
brickdnn.com is, e.g. at the registrar or in Hetzner's DNS console):

- **DKIM** (signs the e-mails): one or more `TXT` or `CNAME` records such as
  `brevo1._domainkey` or `mailjet._domainkey`, exactly as shown.
- **SPF** (which servers may send for the domain): one `TXT` record on the domain itself,
  e.g. `v=spf1 include:spf.brevo.com ~all` or `v=spf1 include:spf.mailjet.com ~all`. There
  must be only **one** SPF record: if one exists, add the `include:` to it. Some providers
  (Postmark, SES) use their own return path instead (a `CNAME` such as `pm-bounces`, or a
  custom MAIL FROM domain); follow their instructions.
- **DMARC** (what receivers do with e-mails that fail the checks): a `TXT` record on
  `_dmarc.brickdnn.com`, to start with
  `v=DMARC1; p=none; rua=mailto:dmarc@brickdnn.com` (reports to an address of yours), after a
  few weeks without problems `p=quarantine`.
- Some providers also want a verification `TXT` record (e.g. `brevo-code:…`).

Wait until the dashboard shows the domain as verified (minutes to hours).

### 2. Settings

| Variable | Where | Example |
| --- | --- | --- |
| `BRICKDNN_EMAIL_HOST` | site settings or `.env` | `smtp-relay.brevo.com` |
| `BRICKDNN_EMAIL_PORT` | site settings or `.env` | `587` (default; `465` switches to TLS from the start) |
| `BRICKDNN_EMAIL_FROM` | site settings or `.env` | `Brick DNN <no-reply@brickdnn.com>`: an address of the verified domain |
| `BRICKDNN_EMAIL_REPLY_TO` | site settings or `.env`, optional | `support@brickdnn.com`: where answers go (`no-reply@` is not read) |
| `BRICKDNN_PUBLIC_URL` | site settings or `.env` | `https://brickdnn.com`: links in e-mails; the clean-up has no request to take it from |
| `BRICKDNN_EMAIL_USER` | **`.env`** | the SMTP login (for some providers it is a key) |
| `BRICKDNN_EMAIL_PASSWORD` | **`.env` only** | the SMTP password or key (secret; ignored in the site settings) |
| `BRICKDNN_ADMIN_EMAIL` | **`.env`** | your address for server errors (keep it out of the public repository) |
| `BRICKDNN_EMAIL_USE_TLS` / `_USE_SSL` | optional | STARTTLS is used on 587, TLS on 465 |
| `BRICKDNN_EMAIL_TIMEOUT` | optional, `10` | seconds to wait for the SMTP server |
| `BRICKDNN_EMAIL_BACKGROUND` | optional, `true` | `false`: send within the request |
| `BRICKDNN_INACTIVE_ACCOUNT_WARNING_DAYS` | optional, `14,3` | days before the deletion of an inactive account its owner is warned |
| `BRICKDNN_SITE_NAME` | optional, `Brick DNN` | the name in e-mails and in authenticator apps |

"Site settings" is `deploy/sites/<domain>/settings.env` in the repository (public!), `.env`
is `/opt/brick-dnn/.env` on the server. Instead of editing `.env`, all of them (the secrets
included) can be GitHub repository variables and secrets, which every deploy writes into
`.env` ([Settings from GitHub](../deploy/server/README.md#settings-from-github)). For brickdnn.com the public lines are prepared
(commented out) in its site settings: remove the `#` and deploy. Before that, update
`deploy/sites/brickdnn.com/legal/privacy.md`: it still says the app sends no e-mails. Take the
e-mail paragraphs from `deploy/legal/privacy.example.md` (section 5, the error reports in
section 4 and the reminder in section 11) and fill in the provider. The secrets go into the
server's `.env`:

```bash
ssh root@SERVER
cd /opt/brick-dnn
nano .env      # add the three lines below, save with Ctrl+O, Enter, Ctrl+X
#   BRICKDNN_EMAIL_USER=…
#   BRICKDNN_EMAIL_PASSWORD=…
#   BRICKDNN_ADMIN_EMAIL=you@example.com
C="docker compose -f docker-compose.yml -f deploy/server/compose.yml"
$C up -d       # restarts the backend and the clean-up with the new settings
$C exec backend python manage.py send_test_email you@example.com
```

`send_test_email` prints the settings in use (never the password), connects to the SMTP
server and sends a test e-mail; it says what is wrong if it cannot. Check that the test
e-mail is not in the spam folder and that its headers say `dkim=pass`, `spf=pass` and
`dmarc=pass` (Gmail: *Show original*).

**Trying it out without a provider** (development): `BRICKDNN_EMAIL_BACKEND=console` prints
e-mails to the log, `BRICKDNN_EMAIL_BACKEND=file` writes them to `BRICKDNN_EMAIL_FILE_PATH`
(default `emails/` in the data directory); both count as "e-mail is set up". Without any
e-mail settings and with `DJANGO_DEBUG=true`, the e-mails that would have been sent are
logged, links included.

## Two-factor authentication

Every account can turn it on under **Account → Two-factor authentication**: password, then a
QR code for an authenticator app (Google Authenticator, Microsoft Authenticator, 1Password,
Aegis, FreeOTP …; time-based codes, 6 digits, 30 seconds) and one code from the app to
confirm. The page then shows **ten recovery codes**, once: each logs in once instead of a
code from the app (e.g. after losing the phone). Turning it on logs out all other devices.

From then on, logging in to the web app, to the OAuth consent page of AI assistants and to
the Django admin asks for a code after the password. API tokens and connected apps keep
working: they were created after such a login. A password reset does not turn it off.
Turning it off, or making new recovery codes, needs the password and a code.

How it is protected: codes of one time step before or after are accepted (clocks drift), a
code works only once, and after five wrong codes the account takes none for 15 minutes (the
second step is also limited per IP address). The secret is stored encrypted with a key
derived from `DJANGO_SECRET_KEY` and never shown again after the setup; recovery codes are
stored as hashes. If `DJANGO_SECRET_KEY` changes, put the old one into Django's
`SECRET_KEY_FALLBACKS`, or authenticator codes stop working (recovery codes still do).

**Lost the phone and the recovery codes?** An administrator turns it off for the account:
Django admin → *Two-factor authentication* → delete the entry, or
`docker compose exec backend python manage.py reset_two_factor <username>`. Make sure the
request really comes from the account's owner. The owner gets a notice by e-mail.

### For the owner of a server: protect the admin account

The accounts in `BRICKDNN_ADMIN_USERS` are superusers: they can see and change all data.

1. Log in to the web app with the admin account, open *Account → Two-factor authentication*,
   turn it on and keep the recovery codes in your password manager (not only on the phone).
2. Log out and in again, and open `/admin/`: the login page there has a field *Two-factor
   code*; check that it works.
3. Then require it for all staff accounts: in `deploy/sites/<domain>/settings.env` set
   `BRICKDNN_REQUIRE_2FA_FOR_STAFF=true` and deploy. Staff accounts without two-factor
   authentication can then still use the web app, but `/admin/` tells them to set it up first
   (and sessions of such accounts that are open lose access).

It is off by default so that a deploy locks nobody out.

## API

The endpoints are in [api.md](api.md#e-mail-password-reset-and-two-factor-authentication).
