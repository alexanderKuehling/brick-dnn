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

# Settings from GitHub. The *Deploy server* workflow sends the repository variables and secrets
# that are set on stdin: a line "brickdnn-settings 1", then one line NAME=<value in base64> each.
# Only the names below are taken, anything else stops the deploy before something changes. They
# replace the same names in .env; a name that is not sent keeps its value there. By hand
# (ssh -n, or brickdnn-deploy on the server) nothing is sent and .env stays as it is.
SETTINGS_FROM_GITHUB=(
  BRICKDNN_PADDLE_ENV BRICKDNN_PADDLE_CLIENT_TOKEN BRICKDNN_PADDLE_PRICES
  BRICKDNN_PADDLE_OVERAGE_PRODUCT BRICKDNN_PADDLE_API_KEY BRICKDNN_PADDLE_WEBHOOK_SECRET
  BRICKDNN_EMAIL_HOST BRICKDNN_EMAIL_PORT BRICKDNN_EMAIL_USER BRICKDNN_EMAIL_PASSWORD
  BRICKDNN_EMAIL_FROM BRICKDNN_EMAIL_REPLY_TO BRICKDNN_ADMIN_EMAIL BRICKDNN_PUBLIC_URL
  BRICKDNN_REQUIRE_2FA_FOR_STAFF
  BRICKDNN_SENTRY_DSN BRICKDNN_SENTRY_FRONTEND_DSN BRICKDNN_SENTRY_ENVIRONMENT
  BRICKDNN_SENTRY_TRACES_SAMPLE_RATE
)
declare -A settings=()
if [ ! -t 0 ] && IFS= read -r -t 10 header && [ "$header" = "brickdnn-settings 1" ]; then
  while IFS= read -r -t 10 line; do
    [ -n "$line" ] || continue
    name="${line%%=*}"
    if [[ ! " ${SETTINGS_FROM_GITHUB[*]} " == *" $name "* ]]; then
      echo "Setting $name may not be set from GitHub; nothing deployed." >&2
      exit 2
    fi
    if ! value="$(printf '%s' "${line#*=}" | base64 -d 2>/dev/null)"; then
      echo "Setting $name is not valid base64; nothing deployed." >&2
      exit 2
    fi
    value="${value%$'\n'}"
    value="${value%$'\r'}"
    # Written in single quotes, which docker compose takes literally (no $ expansion).
    if [[ "$value" == *$'\n'* || "$value" == *$'\r'* || "$value" == *"'"* ]]; then
      echo "Setting $name contains a line break or a ' ; nothing deployed." >&2
      exit 2
    fi
    settings[$name]="$value"
  done
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
# The deployed commit, so that Sentry ties every error to the release it came from.
sed -i '/^BRICKDNN_SENTRY_RELEASE=/d' .env
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
BRICKDNN_SENTRY_RELEASE=brick-dnn@$SHA
EOF

if [ "${#settings[@]}" -gt 0 ]; then
  sed -i '/^# Settings from GitHub/d' .env
  echo "# Settings from GitHub (repository variables and secrets), written by brickdnn-deploy." >> .env
  for name in "${SETTINGS_FROM_GITHUB[@]}"; do
    [ -n "${settings[$name]+set}" ] || continue
    sed -i "/^${name}=/d" .env
    printf "%s='%s'\n" "$name" "${settings[$name]}" >> .env
  done
  echo "Settings from GitHub: ${!settings[*]}"
fi

docker compose -f docker-compose.yml -f deploy/server/compose.yml up -d --build --remove-orphans
docker image prune --force >/dev/null
echo "Deployed $SHA."
