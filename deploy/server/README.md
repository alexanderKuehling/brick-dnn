# Deploying to a server

Backend and frontend run with Docker Compose on one small Linux server (for example a Hetzner
Cloud CX23 with Ubuntu 24.04). [Caddy](https://caddyserver.com) in front of them gets the HTTPS
certificate from Let's Encrypt. After the one-time setup, every push to `main` is deployed
automatically once all CI checks have passed (`.github/workflows/deploy-server.yml`).

```text
GitHub Actions ──ssh deploy@SERVER <commit>──▶ brickdnn-deploy: git checkout, docker compose up
Browser ──HTTPS──▶ Caddy :443 ──▶ nginx (frontend) ──/api──▶ Django (backend) ──▶ Supabase / SQLite
                                                   ──/api/ws/──▶ Daphne (realtime) ◀─ Redis ─▶ Django
Trainings: Django queues them in the database ──▶ worker (manage.py training_worker) trains them
```

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
   This copies the line from your local `.env` to the server:

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
  leaves the server.

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
| `realtime` (live editing, Daphne) and `redis` | half a CPU | 256 MB, 64 MB | the same |
| `cleanup` (deletes old data once a day, sleeps in between) | half a CPU | 256 MB | the same |
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
| Files (datasets, runs; SQLite without `DATABASE_URL`) | Docker volume `brick-dnn_brickdnn-data` |
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

The server keeps files on its disk; turn on Hetzner's backups (or back up the Docker volume
`brick-dnn_brickdnn-data`) to keep them safe. Accounts and networks are in the database.

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
| `database` (answer `503`, status `error`) | Django cannot run `SELECT 1`: Supabase paused (free projects pause after a week without use) or unreachable, wrong `DATABASE_URL` | the Supabase dashboard; `$C logs backend` |
| `worker` (status `degraded`) | no training worker sent a heartbeat in the last minute: trainings wait | *Training workers* in the Django admin; `$C ps worker`, `$C logs worker` |
| `disk` (status `degraded`) | less than 1 GB free in the data volume (`BRICKDNN_HEALTH_MIN_FREE_MB`) | `df -h`, `docker system df`; `docker system prune` removes old images |
| front page | Caddy, nginx or the server is down, or the certificate fails | `$C ps`, `$C logs caddy`; the Hetzner console |

Everyone can see the names of the checks and whether they passed (the repository is public, so
the issue says no more than that). The numbers behind them (free space, seconds since the last
heartbeat, the database's error message) are shown to staff accounts: log in to the site as an
administrator and open `https://DOMAIN/api/health/`. For scripts, set a secret
`BRICKDNN_HEALTH_TOKEN=…` in `/opt/brick-dnn/.env` (`$C up -d`) and send it as header:
`curl -H "X-Health-Token: …" https://DOMAIN/api/health/`.
