# Deploying to a server

Backend and frontend run with Docker Compose on one small Linux server (for example a Hetzner
Cloud CX23 with Ubuntu 24.04). [Caddy](https://caddyserver.com) in front of them gets the HTTPS
certificate from Let's Encrypt. After the one-time setup, every push to `main` is deployed
automatically once all CI checks have passed (`.github/workflows/deploy-server.yml`).

```text
GitHub Actions ──ssh deploy@SERVER <commit>──▶ brickdnn-deploy: git checkout, docker compose up
Browser ──HTTPS──▶ Caddy :443 ──▶ nginx (frontend) ──/api──▶ Django (backend) ──▶ Supabase / SQLite
                                                   ──/api/ws/──▶ Daphne (realtime) ◀─ Redis ─▶ Django
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
answers while they run, the containers get different shares of the server:

| Container | Gets |
| --- | --- |
| `backend`: Django with all its trainings | at most 3/4 of the CPUs and of the memory, no swap |
| `realtime` (live editing, Daphne) and `redis` | small fixed limits: half a CPU and 256 MB, 64 MB |
| `cleanup` (deletes old data once a day, sleeps in between) | half a CPU and 256 MB |
| `caddy` and `frontend` (nginx) | the other quarter, shared with the system; the CPU first when it is busy, killed last |

- Every deploy computes the limits from the size of the server, so they follow a resize
  (Hetzner: *Rescale*, then deploy). They are in `/opt/brick-dnn/.env`
  (`BRICKDNN_BACKEND_CPUS`, `BRICKDNN_BACKEND_MEMORY`, `BRICKDNN_BACKEND_THREADS`); the next
  deploy overwrites changes. A CX23 (2 vCPUs, 4 GB) gives the backend 1.5 CPUs and 2.8 GB.
- In the backend, the API comes first: trainings run with a lower CPU priority, and PyTorch
  uses one thread per whole CPU of the backend (on a CX23 one thread, so half a CPU stays
  free for the API).
- A training that needs more memory than the backend has is the first process the kernel
  kills; the run fails with "The training process stopped unexpectedly" and the API keeps
  running. Imports in the Data tab need about 10–15 times the file size in memory: lower
  `BRICKDNN_DATA_MAX_UPLOAD_MB` on a small server. The server's swap is for building images.
- `docker stats` shows what every container uses.

`setup.sh` copies `deploy.sh` to `/usr/local/bin/brickdnn-deploy`, and deploys do not update
that copy. After a change to `deploy.sh` (such as these limits), copy it again before the
change is deployed, or run `bootstrap.sh` again:

```bash
scp deploy/server/deploy.sh root@SERVER:/usr/local/bin/brickdnn-deploy
```

With the old copy, deploys of commits with the limits fail ("required variable
BRICKDNN_BACKEND_CPUS is missing a value") and the site keeps running the previous version:
copy it, then run the deploy again (GitHub → *Actions* → *Deploy server* → *Run workflow*).

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
```

Old data is only deleted automatically if `.env` sets `BRICKDNN_DELETE_UNUSED_UPLOADS_DAYS`
or `BRICKDNN_DELETE_INACTIVE_ACCOUNTS_DAYS` (then `$C up -d`); see
[docs/privacy.md](../../docs/privacy.md). Docker keeps at most 30 MB of logs per container,
and nginx logs shortened IP addresses only.

Servers set up before the legal pages existed: create the folder once with
`ssh root@SERVER install -d -m 755 -o deploy -g deploy /srv/brick-dnn/legal` (the next deploy
would otherwise create it, owned by root).

The server keeps files on its disk; turn on Hetzner's backups (or back up the Docker volume
`brick-dnn_brickdnn-data`) to keep them safe. Accounts and networks are in the database.
