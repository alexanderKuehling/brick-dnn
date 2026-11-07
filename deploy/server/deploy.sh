#!/usr/bin/env bash
# Deploys one commit of main on this server. setup.sh installs this script as
# /usr/local/bin/brickdnn-deploy, the only command the deploy key may run. GitHub Actions calls
# it after CI has passed:
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

docker compose -f docker-compose.yml -f deploy/server/compose.yml up -d --build --remove-orphans
docker image prune --force >/dev/null
echo "Deployed $SHA."
