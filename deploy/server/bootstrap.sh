#!/usr/bin/env bash
# Sets up the server and GitHub for automatic deploys, in one go, from your machine:
#
#   deploy/server/bootstrap.sh SERVER-IP DOMAIN [SSH-KEY]
#   deploy/server/bootstrap.sh 203.0.113.7 brickdnn.com ~/.ssh/hetzner_alexklg_mvp
#
# SSH-KEY is the private key you log in to the server with as root (for example the one you
# gave Hetzner when creating the server). Without it, ssh uses your default keys.
#
# Steps: creates a deploy key for GitHub Actions, runs setup.sh on the server, copies
# DATABASE_URL from your local .env to the server (replacing an older one), stores the deploy
# key and the server address in GitHub, and starts the first deploy. Running it again is
# safe, e.g. after changing the database password. Needs ssh, scp and the GitHub CLI (gh).
set -euo pipefail

SERVER="${1:?usage: deploy/server/bootstrap.sh SERVER-IP DOMAIN [SSH-KEY]}"
DOMAIN="${2:?usage: deploy/server/bootstrap.sh SERVER-IP DOMAIN [SSH-KEY]}"
KEY="${3:-}"
cd "$(dirname "$0")/../.."

# GitHub Actions has no IPv6, and the domain may not point to the server yet: use the IPv4.
if [[ ! "$SERVER" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]]; then
  echo "SERVER-IP must be the IPv4 address of the server, e.g. 203.0.113.7" >&2
  echo "(Hetzner Cloud Console → Servers → your server → Public IPv4)." >&2
  exit 2
fi

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
if [ -n "$KEY" ]; then
  KEY="${KEY/#\~/$HOME}"
  [ -f "$KEY" ] || { echo "SSH key not found: $KEY" >&2; exit 2; }
  SSH_OPTS+=(-i "$KEY" -o IdentitiesOnly=yes)
fi
remote() { ssh "${SSH_OPTS[@]}" "root@$SERVER" "$@"; }

gh auth status >/dev/null 2>&1 || { echo "Log in to GitHub first: gh auth login" >&2; exit 2; }
remote true || { echo "Cannot log in as root@$SERVER with this key." >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== Deploy key for GitHub Actions"
ssh-keygen -q -t ed25519 -N "" -C "github-actions-deploy" -f "$TMP/deploy"

echo "== Server setup (root@$SERVER)"
scp "${SSH_OPTS[@]}" -q -r deploy/server "root@$SERVER:/root/brickdnn-setup"
remote bash /root/brickdnn-setup/setup.sh "$(printf '%q' "$(cat "$TMP/deploy.pub")")" \
  "$(printf '%q' "$DOMAIN")"

echo "== Database"
if grep -q '^DATABASE_URL=.' .env 2>/dev/null; then
  grep '^DATABASE_URL=' .env | remote \
    "sed -i '/^DATABASE_URL=/d' /opt/brick-dnn/.env && cat >> /opt/brick-dnn/.env"
  echo "DATABASE_URL copied from .env to the server."
else
  echo "No DATABASE_URL in .env: the server keeps its database (SQLite) in a Docker volume."
fi

echo "== GitHub: secret DEPLOY_SSH_KEY, variables DEPLOY_HOST, DEPLOY_URL, DEPLOY_KNOWN_HOSTS"
gh secret set DEPLOY_SSH_KEY < "$TMP/deploy"
gh variable set DEPLOY_HOST --body "$SERVER"
gh variable set DEPLOY_URL --body "https://$DOMAIN"
gh variable set DEPLOY_KNOWN_HOSTS --body "$(ssh-keyscan -t ed25519 "$SERVER" 2>/dev/null)"

echo "== First deploy"
if gh workflow view "Deploy server" >/dev/null 2>&1; then
  gh workflow run "Deploy server" --ref main
  echo "Started. Follow it with: gh run watch \$(gh run list --workflow 'Deploy server' -L1 --json databaseId -q '.[0].databaseId')"
else
  echo "The workflow is not on main yet: merge the pull request, then CI deploys by itself."
fi
echo
echo "Afterwards the site is at https://$DOMAIN (the DNS records must point to $SERVER)."
