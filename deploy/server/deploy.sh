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

# Limits from the size of this server, so they follow a resize: Django with all its trainings
# gets at most three quarters of the CPUs and the memory, the rest stays free for Caddy, nginx
# and the system. In .env, so that `docker compose up` by hand uses them too.
cpus="$(nproc)"
mem_mb="$(awk '/^MemTotal:/ { print int($2 / 1024) }' /proc/meminfo)"
sed -i '/^# Limits for the backend/d; /^BRICKDNN_BACKEND_\(CPUS\|MEMORY\|THREADS\)=/d' .env
[ -z "$(tail -c 1 .env)" ] || echo >> .env  # .env did not end with a line break
cat >> .env <<EOF
# Limits for the backend (3/4 of this server), written by brickdnn-deploy on every deploy.
BRICKDNN_BACKEND_CPUS=$(LC_ALL=C awk -v n="$cpus" 'BEGIN { printf "%.2f", n * 0.75 }')
BRICKDNN_BACKEND_MEMORY=$((mem_mb * 3 / 4))m
BRICKDNN_BACKEND_THREADS=$((cpus * 3 / 4 > 0 ? cpus * 3 / 4 : 1))
EOF

docker compose -f docker-compose.yml -f deploy/server/compose.yml up -d --build --remove-orphans
docker image prune --force >/dev/null
echo "Deployed $SHA."
