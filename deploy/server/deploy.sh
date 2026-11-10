#!/usr/bin/env bash
# Deploys one commit of main on this server. setup.sh installs this script as
# /usr/local/bin/brickdnn-deploy, the only command the deploy key may run. The *Deploy server*
# workflow calls it after CI has passed on main:
#
#   ssh deploy@SERVER <full commit sha>
#
# The same works by hand, e.g. to go back to an older commit of main.
set -euo pipefail

APP=/opt/brick-dnn
SHA="${SSH_ORIGINAL_COMMAND:-${1:-}}"
if [[ ! "$SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Usage: ssh deploy@SERVER <full commit sha of main>" >&2
  exit 2
fi

# One deploy at a time.
exec 9>/tmp/brickdnn-deploy.lock
flock 9

cd "$APP"
git fetch --quiet origin main
# Only commits of main: a stolen deploy key cannot deploy other code.
if ! git merge-base --is-ancestor "$SHA" origin/main 2>/dev/null; then
  echo "Commit $SHA is not on main; nothing deployed." >&2
  exit 1
fi
git checkout --quiet --force --detach "$SHA"

# Limits from the size of this server, so they follow a resize. The web backend and the
# training worker share three quarters of the CPUs and the memory; the rest stays free for
# Caddy, nginx and the system. In .env, so that `docker compose up` by hand uses them too.
#   - Worker (the trainings): three quarters of the CPUs, one thread per whole CPU of those,
#     and the shared memory the web backend does not get.
#   - Web backend (API, data preparation, predictions): about a third of the shared memory,
#     at least 1 GB and at most 3 GB (half of it on servers below 4 GB), and a quarter of
#     the CPUs, at least one and at most two.
# BRICKDNN_BACKEND_* are the shared three quarters. Commits from before the worker need them
# (an older commit can still be deployed), and deploy/server/compose.yml falls back to them
# for the worker's CPUs and threads when an older copy of this script wrote only those.
cpus="$(nproc)"
mem_mb="$(awk '/^MemTotal:/ { print int($2 / 1024) }' /proc/meminfo)"
shared_mb=$((mem_mb * 3 / 4))
web_mb=$((shared_mb * 35 / 100))
[ "$web_mb" -ge 1024 ] || web_mb=1024
[ "$web_mb" -le 3072 ] || web_mb=3072
[ "$shared_mb" -ge 2048 ] || web_mb=$((shared_mb / 2))
shared_cpus="$(LC_ALL=C awk -v n="$cpus" 'BEGIN { printf "%.2f", n * 0.75 }')"
web_cpus="$(LC_ALL=C awk -v n="$cpus" 'BEGIN { c = n / 4; if (c < 1) c = 1; if (c > 2) c = 2; printf "%.2f", c }')"
threads=$((cpus * 3 / 4 > 0 ? cpus * 3 / 4 : 1))
sed -i '/^# Limits for the backend/d; /^# Limits of the containers/d' .env
sed -i '/^BRICKDNN_\(BACKEND\|WEB\|WORKER\)_\(CPUS\|MEMORY\|THREADS\)=/d' .env
[ -z "$(tail -c 1 .env)" ] || echo >> .env  # .env did not end with a line break
cat >> .env <<EOF
# Limits of the containers (web backend and worker share 3/4 of this server), written by brickdnn-deploy on every deploy.
BRICKDNN_BACKEND_CPUS=$shared_cpus
BRICKDNN_BACKEND_MEMORY=${shared_mb}m
BRICKDNN_BACKEND_THREADS=$threads
BRICKDNN_WEB_CPUS=$web_cpus
BRICKDNN_WEB_MEMORY=${web_mb}m
BRICKDNN_WORKER_CPUS=$shared_cpus
BRICKDNN_WORKER_MEMORY=$((shared_mb - web_mb))m
BRICKDNN_WORKER_THREADS=$threads
EOF

docker compose -f docker-compose.yml -f deploy/server/compose.yml up -d --build --remove-orphans
docker image prune --force >/dev/null
echo "Deployed $SHA."
