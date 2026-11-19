# Deploying to a server

Backend and frontend run with Docker Compose on one small Linux server (for example a Hetzner
Cloud CX23 with Ubuntu 24.04). [Caddy](https://caddyserver.com) in front of them gets the HTTPS
certificate from Let's Encrypt. After the one-time setup, every push to `main` is deployed
automatically once all CI checks have passed (`.github/workflows/deploy-server.yml`).

```text
GitHub Actions ──ssh deploy@SERVER <commit>──▶ brickdnn-deploy: git checkout, docker compose up
Browser ──HTTPS──▶ Caddy :443 ──▶ nginx (frontend) ──/api──▶ Django (backend) ──▶ PostgreSQL (db) / Supabase / SQLite
                                                   ──/api/ws/──▶ Daphne (realtime) ◀─ Redis ─▶ Django
Trainings: Django queues them in the database ──▶ worker (manage.py training_worker) trains them
```

nginx (`frontend/nginx.conf`) serves the static landing pages at `/` and `/de/`, `robots.txt`,
`sitemap.xml` and `llms.txt`, and the app (`app.html`) for every other page, the editor at
`/app` included. `/?…` is redirected to `/app?…` (old links into the editor), `/de` to `/de/`.
Caddy passes everything on unchanged, so it needs no routes of its own.

## One-time setup

Create the server with your SSH key and point your domain to it (step 1 below). Then one
command does steps 2–6, with the key you log in to the server with:

```bash
deploy/server/bootstrap.sh SERVER-IP example.com ~/.ssh/your_server_key
```

The same by hand, on your machine, in the repository:

1. Create the server with your own SSH key (the one you log in with as `root`). With an own
   domain, point it to the server at your DNS provider: an `A` record for `@` and `www` with
   the server's IPv4 address (and `AAAA` records with its IPv6 address, or none at all).
   Remove other `A`/`AAAA`/`CNAME` records of these names, or certificates fail.
2. Create a key for GitHub Actions. It is only used for deploying:

   ```bash
   ssh-keygen -t ed25519 -N "" -C github-deploy -f /tmp/brickdnn-deploy
   ```

3. Set up the server (replace `SERVER` with its IP address and `example.com` with your
   domain). Without the domain, the site gets `https://<IP with dashes>.sslip.io`:

   ```bash
   scp -r deploy/server root@SERVER:/root/brickdnn-setup
   ssh root@SERVER bash /root/brickdnn-setup/setup.sh "$(cat /tmp/brickdnn-deploy.pub)" example.com
   ```

   At the end it prints the site address and the server's host key.

4. Add the database, if you use PostgreSQL or Supabase (otherwise SQLite in a Docker volume).
   PostgreSQL on the server itself: see [The PostgreSQL service](#the-postgresql-service) (three
   lines in `.env`, before the first deploy). Supabase or another server: this copies the line
   from your local `.env` to the server:

   ```bash
   grep '^DATABASE_URL=' .env | ssh root@SERVER 'cat >> /opt/brick-dnn/.env'
   ```

5. Tell GitHub where to deploy, then delete the key file:

   ```bash
   gh secret set DEPLOY_SSH_KEY < /tmp/brickdnn-deploy
   gh variable set DEPLOY_HOST --body "SERVER"
   gh variable set DEPLOY_URL --body "https://SITE-ADDRESS"
   gh variable set DEPLOY_KNOWN_HOSTS --body "HOST-KEY-LINE-FROM-SETUP"
   rm /tmp/brickdnn-deploy /tmp/brickdnn-deploy.pub
   ```

6. Push to `main`, or start a deploy by hand: GitHub → *Actions* → *Deploy server* →
   *Run workflow*.

7. Add the imprint, privacy policy and terms of use (templates and a checklist in
   [`deploy/legal/`](../legal/README.md)), in one of two ways:

   - in the repository as `deploy/sites/<BRICKDNN_DOMAIN>/legal/{imprint,privacy,terms}.md`
     (see [`deploy/sites/`](../sites/README.md)): they go online with the next deploy;
   - or on the server, where they are online at once and take precedence:

     ```bash
     scp imprint.md privacy.md terms.md root@SERVER:/srv/brick-dnn/legal/
     ```

   The app links each page whose file exists, and with `terms.md` sign-up asks to accept the
   terms. See [docs/privacy.md](../../docs/privacy.md).

## Security

- The firewall lets only SSH, HTTP and HTTPS in; SSH accepts keys only. Security updates are
  installed automatically.
- The deploy key can do one thing: run `brickdnn-deploy`, which only deploys commits that are
  on `main` of the GitHub repository. It cannot open a shell.
- `/opt/brick-dnn/.env` (database password) is readable by `root` and `deploy` only and never
  leaves the server. The [PostgreSQL service](#the-postgresql-service) has no port on the host
  and is reachable only from the app's containers.
- Besides the commit, a deploy may set the [settings from GitHub](#settings-from-github) in
  `.env`, and only those (Paddle, e-mail): whoever has the deploy key could, for example, point
  e-mails to another SMTP server or set the Paddle webhook secret. Keep the key in GitHub's
  secrets only, as for the deploys themselves.

## Settings from GitHub

The settings of paid plans and e-mail can be kept in GitHub instead of editing `.env` on the
server: *Settings → Secrets and variables → Actions* of the repository. Every deploy sends the
ones that are set to the server, which writes them into `/opt/brick-dnn/.env` (replacing the
same lines) before it starts the containers. Change one, then start a deploy (*Actions →
Deploy server → Run workflow*) or push to `main`.

| Variable (tab *Variables*) | Example | Docs |
| --- | --- | --- |
| `BRICKDNN_PADDLE_ENV` | `sandbox`, later `live` | [billing](../../docs/billing.md#settings) |
| `BRICKDNN_PADDLE_CLIENT_TOKEN` | `test_…` / `live_…` | |
| `BRICKDNN_PADDLE_PRICES` | `starter=pri_…,plus=pri_…,pro=pri_…` | |
| `BRICKDNN_PADDLE_PRICES_YEARLY` | `plus=pri_…,pro=pri_…` (optional) | [paying per year](../../docs/billing.md#paying-per-year) |
| `BRICKDNN_PADDLE_OVERAGE_PRODUCT` | `pro_…` | |
| `BRICKDNN_EMAIL_HOST` | `smtp-relay.brevo.com` | [e-mail](../../docs/email.md#2-settings) |
| `BRICKDNN_EMAIL_PORT` | `587` | |
| `BRICKDNN_EMAIL_FROM` | `Brick DNN <no-reply@brickdnn.com>` | |
| `BRICKDNN_EMAIL_REPLY_TO` | `support@brickdnn.com` | |
| `BRICKDNN_PUBLIC_URL` | `https://brickdnn.com` | |
| `BRICKDNN_REQUIRE_2FA_FOR_STAFF` | `true` (only once your account has two-factor login) | |
| `BRICKDNN_SENTRY_DSN` | DSN of the Sentry server project | [monitoring](../../docs/monitoring.md#settings) |
| `BRICKDNN_SENTRY_FRONTEND_DSN` | DSN of the Sentry frontend project | |
| `BRICKDNN_SENTRY_ENVIRONMENT`, `BRICKDNN_SENTRY_TRACES_SAMPLE_RATE` | `production`, `0.1` (the defaults) | |

| Secret (tab *Secrets*) | Is |
| --- | --- |
| `BRICKDNN_PADDLE_API_KEY` | Paddle API key |
| `BRICKDNN_PADDLE_WEBHOOK_SECRET` | secret of the Paddle notification destination |
| `BRICKDNN_EMAIL_USER` | SMTP login |
| `BRICKDNN_EMAIL_PASSWORD` | SMTP password or key |
| `BRICKDNN_ADMIN_EMAIL` | your address for server errors |
| `SENTRY_AUTH_TOKEN` | Sentry token for uploading the app's source maps while the image is built ([monitoring](../../docs/monitoring.md#readable-stack-traces-in-the-browser)) |

- A variable or secret that is **not set leaves `.env` as it is**; deleting one in GitHub does
  not delete it on the server (do that in `.env`). A value set in GitHub wins over the same
  line in `.env` and over the site settings (`deploy/sites/<domain>/settings.env`).
- Values may contain any character except `'` and line breaks (they are written in single
  quotes, which Docker Compose takes literally, `$` included); with one of those the deploy
  stops and nothing changes.
- Other names cannot be set this way: the server only accepts the ones above, and refuses the
  whole deploy for any other name.
- The settings travel on stdin of the SSH connection, base64-encoded, and never appear on a
  command line or in the log of the workflow.

**Names added later** (`SENTRY_AUTH_TOKEN`) are only sent once the server's deploy script
knows them. The workflow first asks it (`ssh deploy@SERVER settings-names`, which prints the
names it takes) and leaves out the others with a warning in the workflow's summary: the
deploy goes on with the rest. A copy of the script from before this question answers with its
usage; the workflow then sends only the names it has always sent. To take the new names,
install the current script once on the server:

```bash
ssh SERVER 'cd /opt/brick-dnn && sudo -u deploy git fetch -q origin main && sudo -u deploy git show origin/main:deploy/server/deploy.sh > /tmp/brickdnn-deploy.new && sudo install -m 755 /tmp/brickdnn-deploy.new /usr/local/bin/brickdnn-deploy'
```

and deploy again. Every deploy also writes `BRICKDNN_SENTRY_RELEASE=brick-dnn@<commit>` into
`.env`.

Even older copies need the same: from before settings from GitHub, they ignore all of them;
from before the Sentry settings, they refuse a deploy that sends one.

## CPU and memory: the site comes first

Trainings can keep all CPUs busy and use a lot of memory. So that the site still loads and
answers while they run, they do not run in the web backend: the backend puts them into a queue
in the database and the `worker` container trains them
([docs/training.md](../../docs/training.md#the-training-queue)). Every container has its own
share of the server:

| Container | CPUs | Memory (no swap) | On a CX23 (2 vCPUs, 4 GB) |
| --- | --- | --- | --- |
| `worker`: the trainings (`manage.py training_worker` and one process per training) | at most 3/4, and the last to get the CPU when the server is busy | what is left of 3/4 of the memory after the backend's share | 1.5 CPUs, 1.8 GB |
| `backend`: Django (API, data preparation, predictions) | a quarter, at least 1 and at most 2; before the worker when both want the CPU | about a third of those 3/4, at least 1 GB and at most 3 GB | 1 CPU, 1 GB |
| `realtime` (live editing, Daphne) and `redis` | half a CPU; `realtime` before the worker when both want the CPU | 256 MB, 64 MB | the same |
| `cleanup` (deletes old data once a day, sleeps in between) | half a CPU | 256 MB | the same |
| `db` (PostgreSQL, only with [the PostgreSQL service](#the-postgresql-service)) | one CPU; before the worker when both want the CPU | 1.5 GB (`BRICKDNN_DB_MEMORY`), `shared_buffers` 512 MB; killed after the worker | lower both on a CX23 |
| `caddy` and `frontend` (nginx) | the CPU first when it is busy | the rest, shared with the system; killed last | about 350 MB |

The memory limits add up to 3.4 GB on a server with 4 GB (3.75 GB usable), as before the
worker existed: backend and worker share what the backend alone had. CPU limits may overlap
(1.5 + 1 + 0.5 + 0.5 on 2 CPUs): CPU time is shared by priority, and the site wins.

- A worker trains **one run at a time** with one thread per whole CPU it has (on a CX23 one
  thread); `BRICKDNN_WORKER_SLOTS=2` in `.env` trains two at once (they share the worker's
  CPUs and memory). More trainings wait in the queue, paid plans first.
- A training that needs more memory than the worker has is stopped by the kernel: the run
  fails with "The training process stopped unexpectedly", and the worker and the site keep
  running. An import in the Data tab or a prediction that needs more memory than the backend
  has fails the same way without touching the trainings. Imports need about 10–15 times the
  file size in memory: lower `BRICKDNN_DATA_MAX_UPLOAD_MB` on a small server (to about 50 with
  the backend's 1 GB). The server's swap is for building images.
- Predictions still run in the backend (a short process with PyTorch per request, at most 30
  seconds): its limits leave room for one at a time.
- `docker stats` shows what every container uses; *Training workers* in the Django admin
  shows the memory limit each worker found.

**Where the limits come from.** Every deploy computes them from the size of the server, so
they follow a resize (Hetzner: *Rescale*, then deploy), and writes them to
`/opt/brick-dnn/.env`; the next deploy overwrites changes:

| Variable | Is | Written by |
| --- | --- | --- |
| `BRICKDNN_BACKEND_CPUS`, `BRICKDNN_BACKEND_MEMORY`, `BRICKDNN_BACKEND_THREADS` | 3/4 of the server (what backend and worker share) | every version of `brickdnn-deploy` |
| `BRICKDNN_WORKER_CPUS`, `BRICKDNN_WORKER_THREADS`, `BRICKDNN_WORKER_MEMORY` | the worker's limits | `brickdnn-deploy` since the worker exists |
| `BRICKDNN_WEB_CPUS`, `BRICKDNN_WEB_MEMORY` | the backend's limits | `brickdnn-deploy` since the worker exists |

`setup.sh` copies `deploy.sh` to `/usr/local/bin/brickdnn-deploy`, and deploys do not update
that copy. **A server that was set up before the worker existed keeps working without a new
copy**: its script writes only the `BRICKDNN_BACKEND_*` values, and `compose.yml` then gives
the worker those CPUs and threads and uses fixed memory limits, 1792 MB for the worker and
1 GB (and one CPU) for the backend, which are the right ones for a server with 4 GB. With the
old copy the memory limits therefore do **not** follow the size of the server: on a larger
server the worker stays at 1.8 GB. To change that, copy the script again (or run
`bootstrap.sh` again) and deploy:

```bash
scp deploy/server/deploy.sh root@SERVER:/usr/local/bin/brickdnn-deploy
```

On a server with 4 GB a new copy changes nothing worth the trouble (1788 instead of 1792 MB).
Without any of the variables a deploy fails ("required variable BRICKDNN_BACKEND_CPUS is
missing a value") and the site keeps running the previous version.

## What a deploy does to trainings

A deploy rebuilds the image and replaces the containers, the worker included. The worker is
asked to stop, takes no new training, gives the running ones 30 seconds to finish
(`BRICKDNN_WORKER_DRAIN_SECONDS`; Docker waits up to 60) and puts the rest **back into the
queue**, in their old place. The new worker starts them again from the beginning as soon as it
is up; in the web app they are "Queued" for a moment with a note that the worker was
restarted. The compute of the interrupted attempt is not charged twice. A training that is
interrupted like this more than three times (`BRICKDNN_WORKER_MAX_RESTARTS`) fails with a
message that says so; start it again when the deploys are through.

Trainings that only wait in the queue are not affected. The web backend restarts within
seconds as before; trainings go on meanwhile, the web app just cannot show their progress for
that moment.

The first deploy with the worker: trainings that the old backend was running end with it;
they are put into the queue and start again in the worker. Deploying an older commit from
before the worker removes the worker again; what waits in the queue then stays there until a
newer commit is deployed.

## On the server

| What | Where |
| --- | --- |
| Deployed code (a checkout of `main`) | `/opt/brick-dnn` |
| Settings | `/opt/brick-dnn/.env` |
| Imprint, privacy policy, terms of use | `/srv/brick-dnn/legal/{imprint,privacy,terms}.md`, else `deploy/sites/<domain>/legal/` of the checkout |
| Files (datasets, runs; SQLite without `DATABASE_URL`), backups | Docker volume `brick-dnn_brickdnn-data` |
| PostgreSQL of the `db` service | Docker volume `brick-dnn_brickdnn-db` |
| HTTPS certificates | Docker volume `brick-dnn_caddy-data` |

```bash
ssh root@SERVER
cd /opt/brick-dnn
C="docker compose -f docker-compose.yml -f deploy/server/compose.yml"
$C ps                          # what runs
$C logs -f --tail 100 backend  # logs
docker stats --no-stream       # CPU and memory per container
nano .env && $C up -d          # change settings
sudo -u deploy brickdnn-deploy <commit>   # deploy another commit of main, e.g. an older one
$C logs cleanup                # what the daily clean-up deleted
$C exec backend python manage.py purge_old_data --dry-run   # what it would delete now
$C logs -f --tail 100 worker   # what the training worker starts, stops and puts back
$C exec backend python manage.py training_queue   # who waits, who trains, the last 7 days
$C logs realtime | grep "Live connection closed"   # why live connections closed (see below)
$C exec db psql -U postgres brickdnn   # SQL on the PostgreSQL service (if it is on)
```

**The training queue** is also in the Django admin, without a shell: `https://DOMAIN/admin/` →
*Training workers* shows the waiting and running trainings (account, plan, priority, waiting
time), the workers (heartbeat, slots, memory limit, current trainings) and, for the last seven
days, how long trainings waited for a slot (median, 95th percentile, per plan) and how busy
the worker was. Long waits while the worker is busy most of the day mean that the server is
too small: rescale it, or see [docs/scaling.md](../../docs/scaling.md). If the web app says
"No training worker is running", look at `$C ps worker` and `$C logs worker`; trainings wait
in the queue until it is back.

Old data is only deleted automatically if `.env` sets `BRICKDNN_DELETE_UNUSED_UPLOADS_DAYS`
or `BRICKDNN_DELETE_INACTIVE_ACCOUNTS_DAYS` (then `$C up -d`); see
[docs/privacy.md](../../docs/privacy.md). Docker keeps at most 30 MB of logs per container,
and nginx logs shortened IP addresses only.

Servers set up before the legal pages existed: create the folder once with
`ssh root@SERVER install -d -m 755 -o deploy -g deploy /srv/brick-dnn/legal` (the next deploy
would otherwise create it, owned by root).

The server keeps files on its disk and accounts and networks are in the database: see
[Backups](#backups) for keeping both safe.

## The PostgreSQL service

The database can run on the server itself: the `db` service of `docker-compose.yml`
(PostgreSQL 17, image `postgres:17.12-trixie`), instead of Supabase or SQLite. The data then
never leaves the server, queries do not travel to another data centre, and the backups below
are native `pg_dump` files. It is **off until `.env` turns it on**, so servers with SQLite or
Supabase are not affected:

| Line in `/opt/brick-dnn/.env` | Does |
| --- | --- |
| `COMPOSE_PROFILES=postgres` | starts the `db` service with every `docker compose up`, so with every deploy |
| `BRICKDNN_POSTGRES_PASSWORD=…` | the password of the database role `brickdnn`, set when the service starts for the first time |
| `DATABASE_URL=postgresql://brickdnn:…@db:5432/brickdnn` | Brick DNN uses it (the same password) |

Why a Compose profile: Docker Compose reads `COMPOSE_PROFILES` from `.env` by itself, so the
deploy script needs no change and no new copy on the server, and `docker compose …` by hand
does the same as a deploy. Backend, worker, realtime and cleanup wait for `db` to be healthy,
but the dependency is `required: false`: without the profile they start as before. These are
lines in `.env`, not in `deploy/sites/<domain>/settings.env`, because Compose reads only `.env`
(the site settings are read by Django) and because two of them hold the password.

- **The password**: `openssl rand -hex 32` (64 characters from `0-9a-f`; base64 would bring
  `/`, `+` and `=`, which would have to be percent-encoded in `DATABASE_URL`). It is only in
  `.env` on the server, never in the repository or in GitHub.
- **Data**: Docker volume `brick-dnn_brickdnn-db`
  (`/var/lib/docker/volumes/brick-dnn_brickdnn-db/_data`), with data checksums. Copies of it
  in a running state are not reliable: the backups take the database with `pg_dump`.
- **Access**: no port on the host. Only backend, worker, realtime and cleanup reach it, over the
  internal Docker network `database`, which has no way out. The role `brickdnn` owns the
  database but is no superuser; the superuser `postgres` has no password and logs in only
  inside the container: `$C exec db psql -U postgres brickdnn`.
- **Memory and CPU** (`deploy/server/compose.yml`): `shared_buffers` 512 MB, the container at
  most 1.5 GB and one CPU, before the trainings when both want the CPU, and among the last
  processes to be killed when memory runs out. On a smaller server set
  `BRICKDNN_DB_SHARED_BUFFERS=128MB` and `BRICKDNN_DB_MEMORY=512m` in `.env`
  (`BRICKDNN_DB_CPUS` for the CPUs), then `$C up -d`.
- **Updates**: a new minor version (17.x) is a new image tag in `docker-compose.yml`, deployed
  like any commit (the database restarts: a few seconds without it). A new major version (18)
  cannot just change the tag: it needs a dump and restore (`manage.py move_database` into a new
  service, or `pg_upgrade`), and the backend image must have `pg_dump` of at least that version
  for the backups (`backend/Dockerfile`).
- **A new password**: changing `BRICKDNN_POSTGRES_PASSWORD` alone does nothing (it is used at the
  first start only). Set the new one in the database (`$C exec db psql -U postgres -c
  '\password brickdnn'` asks for it twice), then in both lines of `.env`, then `$C up -d`.

Moving an existing site onto it: [Moving the database onto the server](#moving-the-database-onto-the-server).

## Backups

What is kept where, and what is backed up:

| Data | Where | Backed up by |
| --- | --- | --- |
| Accounts, networks, runs, usage (the database) | the `db` service, Supabase or another PostgreSQL (`DATABASE_URL`), else SQLite in the data volume | `manage.py backup` (below). Supabase also: **check your plan** (dashboard → *Database* → *Backups*): paid plans keep daily backups for 7 days, the free plan has none you can restore yourself. |
| Uploads, datasets, trained weights, the generated secret key | Docker volume `brick-dnn_brickdnn-data` on the server's disk | nothing, unless you turn on Hetzner's backups; and `manage.py backup` |

**1. Turn on Hetzner's server backups** (the simplest offsite copy): Hetzner Cloud Console →
the server → *Backups* → *Enable backups*. One click, about 20 % of the server's price; Hetzner
copies the whole disk once a day (Docker volumes included) and keeps the last seven. To go
back, *Backups* → a backup → *Restore* (replaces the disk) or *Convert to image* and create a
second server from it to take files from. They do not include a Supabase database, and a copy
of the disk of a running PostgreSQL is not a reliable backup of it: that is what step 2 is for.

**2. `manage.py backup`, daily.** With `BRICKDNN_BACKUP_DIR` set (brickdnn.com sets
`/data/backups` in `deploy/sites/brickdnn.com/settings.env`), the cleanup service runs it once a
day after the clean-up and writes into the data volume a folder `brickdnn-<date and time>`
with

- the database as of one moment: `pg_dump` in its custom format (`database.pgdump`) where the
  installed `pg_dump` is not older than the server (the Docker image has version 17: the `db`
  service and Supabase up to 17), else Django's `dumpdata` in one transaction
  (`database.json.gz`; it loads into PostgreSQL of any version and into SQLite), or SQLite's own
  online backup;
- the files of the data volume as `files.tar.gz` (without the cache and the backups). Above
  `BRICKDNN_BACKUP_MAX_FILES_MB`, or when they would not fit on the disk, only a list of them
  (`files.txt`); `BRICKDNN_BACKUP_FILES=list` or `none` chooses that up front, e.g. when
  Hetzner's backups have the files;
- `manifest.json`: when, which version, what is inside.

It keeps the newest backup of each of the last 7 days and of each of the last 4 weeks
(`BRICKDNN_BACKUP_KEEP_DAILY`, `BRICKDNN_BACKUP_KEEP_WEEKLY`) and deletes the others. The
folder holds password hashes and the secret key: it is readable by the app's user only, and
copies of it should be kept as carefully. `/api/health/` reports `backup: failed` when the last
backup failed or the newest is older than 50 hours (`BRICKDNN_BACKUP_MAX_AGE_HOURS`), so the
uptime check opens an issue. **Mind the disk**: every backup on the server's disk takes space
(3 + 2 backups of up to 1 GB of files on brickdnn.com, which copies them offsite); `$C exec cleanup python manage.py backup
--list` shows their sizes. Without Hetzner's backups these copies are on the same disk as the
site: they help after a mistake or a broken database, not after losing the server.

**3. Hourly backups of the database**, with `BRICKDNN_BACKUP_KEEP_HOURLY` above 0 (e.g. `24`,
in `.env` or the site settings; unset or `0`: none). In the 23 hours between two daily backups
the cleanup service runs `manage.py backup --hourly` every hour: the database alone, in a folder
`brickdnn-<date and time>-hourly` in the same backup folder (so the offsite copy below takes it
too). The newest backup of each of the last 24 hours is kept, in addition to the daily and
weekly ones; the daily ones are never replaced by an hourly one. Each takes as much space as
the database's dump (`backup --list` shows the sizes): 24 of them take 24 times that. With
hourly backups on, `/api/health/` also fails when the last hourly one failed or the newest
backup is older than three hours.

**Offsite copies** are not built in (the image has no storage client), but `rclone` on the
server copies the backup folder to any storage it supports: Infomaniak Swiss Backup and other
OpenStack Swift storage, S3-compatible storage (Hetzner Object Storage, Backblaze B2 …) and
more. brickdnn.com copies to Swiss Backup this way:

1. `sudo apt-get install -y rclone`, then put the provider's `rclone.conf` into
   `/root/.config/rclone/` (Swiss Backup offers it for download per device), readable by root
   only (`chmod 600`), with the password filled in. Keep the password in a password manager as
   well: without it the copies are lost with the server. It does not belong in GitHub or `.env`.
2. `sudo rclone listremotes` names the remote; `sudo rclone mkdir REMOTE:brickdnn-backups`
   creates the container.
3. A first copy, by hand:
   `sudo rclone copy /var/lib/docker/volumes/brick-dnn_brickdnn-data/_data/backups REMOTE:brickdnn-backups -v`.
4. From root's crontab (`sudo crontab -e`), one line. With daily backups only, every six hours:

   ```
   0 */6 * * * rclone copy /var/lib/docker/volumes/brick-dnn_brickdnn-data/_data/backups REMOTE:brickdnn-backups --log-file /var/log/rclone-backup.log && rclone delete REMOTE:brickdnn-backups --min-age 35d --log-file /var/log/rclone-backup.log
   ```

   With hourly backups, every hour at half past (a backup in progress is a hidden folder
   `.partial-…`, which stays out), keeping the hourly ones offsite for three days and the rest
   for 35:

   ```
   30 * * * * rclone copy /var/lib/docker/volumes/brick-dnn_brickdnn-data/_data/backups REMOTE:brickdnn-backups --exclude "/.partial-*/**" --log-file /var/log/rclone-backup.log && rclone delete REMOTE:brickdnn-backups --min-age 35d --log-file /var/log/rclone-backup.log && rclone delete REMOTE:brickdnn-backups --include "/brickdnn-*-hourly/**" --min-age 3d --log-file /var/log/rclone-backup.log
   ```

   Worth it once hourly backups are on: a lost server then costs at most an hour of the
   database instead of up to six, and every run only transfers the new folder.

`copy`, not `sync`: a backup deleted on the server (by the rotation, a mistake or an attacker)
stays in the offsite copy, and `delete --min-age 35d` keeps that store from growing without
end. At fixed times rather than right after each backup, because the daily backup runs ten
minutes after the cleanup service starts and then once a day, so its time moves with every
deploy; only new files are transferred. `sudo tail /var/log/rclone-backup.log` shows the last runs.

```bash
$C exec cleanup python manage.py backup --list   # the backups, newest first
$C exec cleanup python manage.py backup          # one now
$C exec cleanup python manage.py backup --hourly # an hourly one now (database only)
```

**Restore** (replaces the database, and writes the backup's files into the data volume; files
added later stay):

```bash
$C stop backend worker realtime cleanup
$C run --rm --no-deps backend python manage.py restore latest   # or a folder name; asks first
$C up -d
```

`--database-only` and `--files-only` restore one part. `latest` is the newest backup of either
kind; an hourly one has no files (`restore` says so and leaves them alone): for the files take
the newest daily folder, `restore brickdnn-<date and time> --files-only`. A backup of newer code
than the one running restores too, but restore with the same commit (`manifest.json` names the
version) or newer: `restore` runs `migrate` afterwards.

**Restore drill** (once, and after big changes). With the `db` service: restore the newest
backup into a scratch database next to the real one and compare the numbers, without touching
the site:

```bash
B="$(ls -1d /var/lib/docker/volumes/brick-dnn_brickdnn-data/_data/backups/brickdnn-2* | tail -n 1)"
$C exec -T db createdb -U postgres drill
$C exec -T db pg_restore -U postgres --dbname=drill --clean --if-exists --no-owner --exit-on-error < "$B/database.pgdump"
Q="SELECT (SELECT count(*) FROM auth_user) AS accounts, (SELECT count(*) FROM brick_dnn_core_project) AS networks, (SELECT max(applied) FROM django_migrations) AS last_migration"
$C exec -T db psql -U postgres -d drill -c "$Q"      # the backup
$C exec -T db psql -U postgres -d brickdnn -c "$Q"   # the site now: the same or a little more
$C exec -T db dropdb -U postgres drill
```

With SQLite or a `database.json.gz`: copy a backup to your computer and restore it into an
empty SQLite database, which proves it is complete without touching the site:

```bash
scp -r root@SERVER:/var/lib/docker/volumes/brick-dnn_brickdnn-data/_data/backups/brickdnn-2026… /tmp/drill-backup
cd backend
DATABASE_URL= BRICKDNN_DATA_DIR=/tmp/drill .venv/bin/python manage.py restore /tmp/drill-backup --no-input
DATABASE_URL= BRICKDNN_DATA_DIR=/tmp/drill .venv/bin/python manage.py runserver   # log in, look around
rm -rf /tmp/drill /tmp/drill-backup   # it holds password hashes
```

## Moving the database onto the server

From Supabase (or any other PostgreSQL) to the [PostgreSQL service](#the-postgresql-service)
on the server itself, with `manage.py move_database`. It copies only Brick DNN's tables (none of
Supabase's own schemas, roles, owners or grants), resets the sequences, runs `migrate --check`
and verifies the copy: the same row count and a checksum of every row in every table, sequences
ahead of the highest id. It prints a line per table and `RESULT: PASS` or `FAIL`. It never
changes the source, which stays as it is for going back. All commands run on the server as
root, in `/opt/brick-dnn`.

**Expected downtime: about 5 minutes** (the site answers again after 1–3 of them; the rest are
the checks), on a database of up to some hundred MB. The rehearsal (step 2) prints how long the
dump and the restore took; the real move takes about as long. Plan a window of 30 minutes at a
quiet time and tell the users beforehand: during the window the app's API answers with
errors (the landing pages stay up), live editing reconnects afterwards, and trainings that run
are put back into the queue and start again after the move. If a check of the uptime workflow
falls into the window, it opens an outage issue and closes it when the site is back.

### 0. Prerequisites

```bash
ssh root@SERVER
cd /opt/brick-dnn
C="docker compose -f docker-compose.yml -f deploy/server/compose.yml"
$C exec backend pg_dump --version      # 17.x: the commit with the db service is deployed
$C exec backend sh -c 'psql "$DATABASE_URL" -tAc "SHOW server_version" -c "SELECT pg_size_pretty(pg_database_size(current_database()))"'
df -h /var/lib/docker                  # free space: a few times the database's size
```

- Supabase's version must be **17 or lower** (it is 15 or 17); with 18 the backend's
  `pg_dump` cannot dump it.
- `DATABASE_URL` in `.env` points to Supabase's **session pooler** (port 5432), as
  [docs/database.md](../../docs/database.md#supabase) recommends. The transaction pooler
  (6543) does not work for `pg_dump`: copy the session pooler's string from the dashboard
  (*Connect*) first.
- A fresh backup of Supabase, just in case: `$C exec cleanup python manage.py backup`.

### 1. Switch the PostgreSQL service on (the site keeps running on Supabase)

```bash
cp -p .env /root/env-before-db-move && chmod 600 /root/env-before-db-move   # for going back
[ -z "$(tail -c 1 .env)" ] || echo >> .env   # .env ends with a line break
PW="$(openssl rand -hex 32)"
printf '%s\n' '# PostgreSQL on this server (deploy/server/README.md, "The PostgreSQL service").' \
  "COMPOSE_PROFILES=postgres" "BRICKDNN_POSTGRES_PASSWORD=$PW" >> .env
unset PW
$C up -d --wait db                     # the first start creates the role and the database
$C ps db                               # "healthy"
```

### 2. Rehearse (as often as you like, while the site runs)

```bash
export LOCAL_DATABASE_URL="postgresql://brickdnn:$(sed -n 's/^BRICKDNN_POSTGRES_PASSWORD=//p' .env)@db:5432/brickdnn"
$C run --rm --no-deps -e LOCAL_DATABASE_URL backend python manage.py move_database --to env:LOCAL_DATABASE_URL --dry-run
```

It copies from `DATABASE_URL` (Supabase) into the new database, verifies, prints the report and
drops the copy again. The numbers are those of one moment of Supabase (a snapshot that the dump
shares), so writes while it runs do not spoil the comparison. It must end with `RESULT: PASS`.
`--from` and `--to` take a URL or `env:NAME` (the URL in an environment variable: the password
stays off the command line); `--keep` leaves the copy to look at (`$C exec db psql -U postgres
brickdnn`), and the next run then needs `--replace`. Steps 4 and 5 need `C` and
`LOCAL_DATABASE_URL` of this shell: in a new one, run the first lines of step 0 and step 2 again.

### 3. The window: stop all writes

There is no maintenance mode: stopping the services that write is what makes the copy
complete. Look first whether a training runs (`$C exec backend python manage.py
training_queue`); a running one is put back into the queue and starts again after the move.

```bash
$C stop worker cleanup realtime backend    # the worker takes up to a minute (it puts trainings back)
$C ps                                      # only db, redis, frontend, caddy run
```

### 4. Move and verify

```bash
$C run --rm --no-deps -e LOCAL_DATABASE_URL backend python manage.py move_database --to env:LOCAL_DATABASE_URL 2>&1 | tee /root/db-move.log
```

The last lines must say `PASS  source unchanged` and `RESULT: PASS`. `FAIL` (or an error)
means: do not switch. The site is back as before with `$C up -d`; look at the report, fix the
cause and run step 4 again with `--replace` (it drops the incomplete copy, after asking).

### 5. Switch `DATABASE_URL` and start

```bash
sed -i '/^DATABASE_URL=/d' .env
echo "DATABASE_URL=$LOCAL_DATABASE_URL" >> .env
unset LOCAL_DATABASE_URL
grep -c '^DATABASE_URL=postgresql://brickdnn:' .env   # 1
$C up -d
```

### 6. Smoke checks

```bash
$C ps                                                    # all running, healthy after a minute
$C exec backend sh -c 'echo "$DATABASE_URL" | sed "s/:[^:@]*@/:…@/"'   # …@db:5432/brickdnn
curl -s https://DOMAIN/api/health/                       # "status": "ok"
$C exec cleanup python manage.py backup                  # "database postgresql (pgdump)"
```

In the browser: log in, open the list of networks and one network (the editor shows it),
start a tiny training (a small example with a built-in dataset, one epoch; `$C logs -f worker`
shows it start and finish), and open a network in two tabs to see live editing. In the Django
admin, *Training workers* shows the worker's heartbeat. If something is wrong, go back (step 7).

Afterwards:

- Merge the pull request that switches brickdnn.com's privacy policy and settings to the new
  database (hourly backups, `BRICKDNN_BACKUP_KEEP_HOURLY=24`); the next deploy takes it.
- Change the rclone line in root's crontab to the hourly one (Backups, step 4 of the offsite
  copies).
- Do the restore drill above once with the first backups of the new database.

### 7. Going back to Supabase

Supabase was not changed. Within the window, nothing is lost:

```bash
$C stop worker cleanup realtime backend
sed -i '/^DATABASE_URL=/d' .env
grep '^DATABASE_URL=' /root/env-before-db-move >> .env
$C up -d
```

After the site has run on the new database for a while, Supabase lacks what was written since.
To keep it, copy it back before switching (with the app stopped):
`$C run --rm --no-deps -e SUPABASE_URL backend python manage.py move_database --from env:LOCAL_DATABASE_URL --to env:SUPABASE_URL --replace`
(both variables exported first; `--replace` overwrites Brick DNN's tables at Supabase, which
are protected from its Data API again afterwards).

### 8. After a few days: say goodbye to Supabase

When the site has run well on the new database for a few days (and the daily and hourly
backups work, `$C exec cleanup python manage.py backup --list`):

1. A last copy of Supabase, kept out of the rotation:

   ```bash
   export SUPABASE_URL="$(sed -n 's/^DATABASE_URL=//p' /root/env-before-db-move)"
   $C run --rm --no-deps -e SUPABASE_URL backend sh -c 'pg_dump --format=custom --no-owner --no-privileges --schema=public --file=/data/backups/supabase-final.pgdump "$SUPABASE_URL" && pg_restore --list /data/backups/supabase-final.pgdump | grep -c "TABLE DATA"'
   unset SUPABASE_URL
   scp root@SERVER:/var/lib/docker/volumes/brick-dnn_brickdnn-data/_data/backups/supabase-final.pgdump .   # on your computer
   ```

   It holds password hashes: keep it encrypted, and delete it (on the server:
   `rm /var/lib/docker/volumes/brick-dnn_brickdnn-data/_data/backups/supabase-final.pgdump`) when
   you no longer need it; the offsite copy goes after 35 days.
2. Delete the Supabase project: dashboard → the project → *Project Settings* → *General* →
   *Delete project*. Then `shred -u /root/env-before-db-move /root/db-move.log`.
3. Paperwork: note in your records of processing (Art. 30 GDPR) and the list of data
   processing agreements that the Supabase DPA ended with the deletion on that date (keep the
   signed copy as evidence), and that the database is now covered by Infomaniak's agreement for
   the server.

## Monitoring

GitHub checks the site every 15 minutes, without any other service or account
(`.github/workflows/uptime.yml`, script `deploy/server/uptime.py`):

- `DEPLOY_URL/api/health/` must answer `"status": "ok"` and the front page must load. A failed
  check is tried twice more, 30 seconds apart, so a deploy does not count as an outage.
- When the site is down, the workflow opens **one issue** labelled `outage` ("brickdnn.com is
  down") with the failed checks. While the outage lasts it comments on that issue when the
  problem changes and every six hours, never opening a second one. When the site is back it
  comments how long the outage took and closes the issue.
- GitHub sends you an e-mail and a notification in the GitHub app for the new issue and its
  comments, **if you watch the repository**: on the repository page, *Watch* → *All Activity*
  (or *Custom* → *Issues*), and in your GitHub settings under *Notifications* have e-mail (and
  the mobile app's push notifications) on for *Watching*. Test it once: *Actions* → *Uptime* →
  *Run workflow*.
- It needs only the variable `DEPLOY_URL` (set for deploys); without it nothing runs. Public
  repositories pay nothing for these runs. GitHub may start scheduled runs some minutes late,
  and **switches schedules off after 60 days without any commit** to the repository: GitHub
  says so by e-mail, and *Actions* → *Uptime* → *Enable workflow* turns it on again.

What the checks mean:

| Check | Fails when | Look at |
| --- | --- | --- |
| `database` (answer `503`, status `error`) | Django cannot run `SELECT 1`: the `db` service down, Supabase paused (free projects pause after a week without use) or unreachable, wrong `DATABASE_URL` | `$C ps db`, `$C logs db`; the Supabase dashboard; `$C logs backend` |
| `worker` (status `degraded`) | no training worker sent a heartbeat in the last minute: trainings wait | *Training workers* in the Django admin; `$C ps worker`, `$C logs worker` |
| `disk` (status `degraded`) | less than 1 GB free in the data volume (`BRICKDNN_HEALTH_MIN_FREE_MB`) | `df -h`, `docker system df`; `docker system prune` removes old images |
| front page | Caddy, nginx or the server is down, or the certificate fails | `$C ps`, `$C logs caddy`; the Hetzner console |

Everyone can see the names of the checks and whether they passed (the repository is public, so
the issue says no more than that). The numbers behind them (free space, seconds since the last
heartbeat, the database's error message) are shown to staff accounts: log in to the site as an
administrator and open `https://DOMAIN/api/health/`. For scripts, set a secret
`BRICKDNN_HEALTH_TOKEN=…` in `/opt/brick-dnn/.env` (`$C up -d`) and send it as header:
`curl -H "X-Health-Token: …" https://DOMAIN/api/health/`.

## Troubleshooting: live editing drops

The live badge of the builder only says *Offline* after a connection has stayed down for 8
seconds; shorter drops are reconnected without notice. Every closed connection leaves one line
in the log of the `realtime` container, with the close code, why it closed and how long it was
open (when the server closed it, the reason says so and names the code it sent):

```bash
$C logs -t --since 24h realtime | grep "Live connection closed"
# … Live connection closed: project 12, user 3, code 1006 (connection lost: network, proxy or ping timeout), after 1312 s, 410 messages
# … Live connection closed: project 12, user 5, code 1001 (going away: page closed or reloaded), after 95 s, 31 messages
# … Live connection closed: project 12, user 3, code 1011 (closed by the server with 1011: server error (OperationalError)), after 40 s, 12 messages

# How often each kind happened (the part in brackets):
$C logs --since 24h realtime | grep -o "code [0-9]* ([^)]*" | sort | uniq -c | sort -rn
# Errors behind "server error" (also in Sentry, if it is set up):
$C logs --since 24h realtime | grep -B2 -A20 "Live editing: error"
# Restarts and out-of-memory kills of the realtime container:
docker inspect -f '{{.State.StartedAt}} restarts={{.RestartCount}} oom={{.State.OOMKilled}}' $($C ps -q realtime)
docker stats --no-stream $($C ps -q realtime)
```

In the browser, the developer tools' console shows the client's side at the *Verbose* level:
`[live] connection closed {code, reason, wasClean, seconds}` and `[live] reconnecting: …`
(e.g. after a missed change, or when the page came back from the background and the
connection did not answer).

What the codes mean:

| Code in the log | Cause | What to do |
| --- | --- | --- |
| 1000 after about 86400 s | Daphne ends every connection after a day (`--websocket_timeout`) | nothing; the client reconnects |
| 1001 | the tab was closed or reloaded | nothing |
| 1006 *connection lost* | network gone (phones, Wi-Fi changes), a tab suspended in the background that no longer answered Daphne's ping (after 30 s of silence plus 30 s), or a proxy closed it | many at the same second: a restart of `frontend`, `realtime` or `caddy` (compare with `$C ps` and the times of deploys) |
| *closed by the server with 1011: server error* | an error in the consumer, mostly the database (unreachable, SQLite locked by a training) or Redis | the traceback before the line; `$C logs redis` |
| *with 4401* | no or an expired login (the client refreshes its token once) | many: the clocks of server and clients, `SIMPLE_JWT` lifetimes |
| *with 4429: more than 30 messages a second* | a client sent too much (the web app sends at most about 20 a second) | a client bug or abuse |
| *with 4403 / 4404* | access removed, project deleted | nothing |

Further causes and what is set up against them:

- **Deploys.** Every deploy builds a new frontend image (the release is built into the app) and
  so restarts nginx, and with backend changes also `realtime`: every live connection drops
  once and reconnects within seconds. A day with many deploys looks like drops every few
  minutes in the log, all at the same times.
- **Reloads of Caddy** (`caddy reload`, a changed Caddyfile) keep WebSockets open for up to 5
  more minutes (`stream_close_delay`); a restart of the `caddy` container closes them at once.
  Certificate renewals close nothing.
- **nginx** passes `/api/ws/` on with HTTP/1.1 and the `Upgrade`/`Connection` headers, without
  buffering (upgraded connections are passed through), with timeouts of an hour; clients ping
  every 25 seconds, so these timeouts never end a working connection. It looks the address of
  `realtime` up again at least every 10 seconds, so reconnects reach a `realtime` container
  that was restarted on its own.
- **A busy server.** `realtime` has half a CPU and 256 MB (it uses about 120 MB) and gets the
  CPU before the trainings, like the backend. The login (auth message) has 10 seconds to
  arrive; the database work after it no longer counts, so a slow database after a deploy,
  when every client logs in again at once, does not close connections with 4401.
- **The channel layer** (Redis) keeps at most 500 messages per connection for 30 seconds. A
  client that falls behind loses messages, notices the gap in the versions and loads the
  project again (`[live] reconnecting: missed a change`); the `realtime` log says
  `… channels over capacity in group project-…` at the same time.
