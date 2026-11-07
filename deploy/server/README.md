# Deploying to a server

Backend and frontend run with Docker Compose on one small Linux server (for example a Hetzner
Cloud CX23 with Ubuntu 24.04). [Caddy](https://caddyserver.com) in front of them gets the HTTPS
certificate from Let's Encrypt. After the one-time setup, every push to `main` is deployed
automatically once all CI checks have passed (job `deploy-server` in `.github/workflows/ci.yml`).

```
GitHub Actions ──ssh deploy@SERVER <commit>──▶ brickdnn-deploy: git checkout, docker compose up
Browser ──HTTPS──▶ Caddy :443 ──▶ nginx (frontend) ──/api──▶ Django (backend) ──▶ Supabase / SQLite
```

## One-time setup

On your machine, in the repository:

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

6. Push to `main`, or start a deploy by hand: GitHub → *Actions* → *CI* → *Run workflow*.

## Security

- The firewall lets only SSH, HTTP and HTTPS in; SSH accepts keys only. Security updates are
  installed automatically.
- The deploy key can do one thing: run `brickdnn-deploy`, which only deploys commits that are
  on `main` of the GitHub repository. It cannot open a shell.
- `/opt/brick-dnn/.env` (database password) is readable by `root` and `deploy` only and never
  leaves the server.

## On the server

| What | Where |
| --- | --- |
| Deployed code (a checkout of `main`) | `/opt/brick-dnn` |
| Settings | `/opt/brick-dnn/.env` |
| Files (datasets, runs; SQLite without `DATABASE_URL`) | Docker volume `brick-dnn_brickdnn-data` |
| HTTPS certificates | Docker volume `brick-dnn_caddy-data` |

```bash
ssh root@SERVER
cd /opt/brick-dnn
C="docker compose -f docker-compose.yml -f deploy/server/compose.yml"
$C ps                          # what runs
$C logs -f --tail 100 backend  # logs
nano .env && $C up -d          # change settings
sudo -u deploy brickdnn-deploy <commit>   # deploy another commit of main, e.g. an older one
```

The server keeps files on its disk; turn on Hetzner's backups (or back up the Docker volume
`brick-dnn_brickdnn-data`) to keep them safe. Accounts and networks are in the database.
