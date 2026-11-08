#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu or Debian server (e.g. Hetzner Cloud CX23) for Brick DNN.
# Run it as root, from the repository on your machine:
#
#   scp -r deploy/server root@SERVER:/root/brickdnn-setup
#   ssh root@SERVER bash /root/brickdnn-setup/setup.sh "ssh-ed25519 AAAA… github-deploy" [DOMAIN]
#
# The first argument is the public key GitHub Actions deploys with. Without DOMAIN the site
# gets <IP>.sslip.io (works without an own domain). The script installs Docker, a firewall
# (SSH, HTTP, HTTPS), automatic security updates and swap, clones the repository to
# /opt/brick-dnn, writes /opt/brick-dnn/.env (if missing) and creates the user "deploy",
# whose key can only deploy commits of main (deploy.sh). Running it again is safe.
set -euo pipefail

DEPLOY_KEY="${1:?usage: setup.sh 'DEPLOY_PUBLIC_KEY' [DOMAIN]}"
DOMAIN="${2:-}"
REPO="${BRICKDNN_REPO:-https://github.com/alexanderKuehling/brick-dnn.git}"
APP=/opt/brick-dnn
HERE="$(cd "$(dirname "$0")" && pwd)"
export DEBIAN_FRONTEND=noninteractive

if [[ ! "$DEPLOY_KEY" =~ ^ssh-(ed25519|rsa)\ [A-Za-z0-9+/=]+( .*)?$ ]]; then
  echo "The first argument must be a public SSH key (ssh-ed25519 AAAA…)." >&2
  exit 2
fi

echo "== Packages, automatic security updates"
apt-get update -q
apt-get upgrade -yq
apt-get install -yq ca-certificates curl git ufw unattended-upgrades
cat > /etc/apt/apt.conf.d/20auto-upgrades <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
EOF

echo "== Firewall: only SSH, HTTP and HTTPS"
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443
ufw --force enable

echo "== Swap (for building images on 4 GB RAM; the backend itself does not swap)"
if [ -z "$(swapon --show)" ]; then
  fallocate -l 4G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "== Docker"
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh

echo "== SSH: log in with keys only"
# sshd uses the first value it reads; this file comes before cloud-init's 50-cloud-init.conf.
printf 'PasswordAuthentication no\nKbdInteractiveAuthentication no\n' \
  > /etc/ssh/sshd_config.d/10-brickdnn.conf
systemctl reload ssh 2>/dev/null || systemctl reload sshd 2>/dev/null || true

echo "== User deploy: its key can only run brickdnn-deploy"
id deploy >/dev/null 2>&1 || useradd --create-home --shell /bin/bash deploy
usermod -aG docker deploy
install -m 755 "$HERE/deploy.sh" /usr/local/bin/brickdnn-deploy
install -d -m 700 -o deploy -g deploy /home/deploy/.ssh
echo "command=\"/usr/local/bin/brickdnn-deploy\",restrict $DEPLOY_KEY" \
  > /home/deploy/.ssh/authorized_keys
chown deploy:deploy /home/deploy/.ssh/authorized_keys
chmod 600 /home/deploy/.ssh/authorized_keys

echo "== Code in $APP"
[ -d "$APP/.git" ] || git clone --quiet "$REPO" "$APP"
chown -R deploy:deploy "$APP"

echo "== Settings in $APP/.env"
if [ ! -f "$APP/.env" ]; then
  if [ -z "$DOMAIN" ]; then
    IP="$(ip -4 route get 1.1.1.1 | sed -n 's/.* src \([0-9.]*\).*/\1/p')"
    DOMAIN="$(echo "$IP" | tr . -).sslip.io"
  fi
  cat > "$APP/.env" <<EOF
# Brick DNN on this server, written by deploy/server/setup.sh. Docker Compose reads it.
BRICKDNN_DOMAIN=$DOMAIN
DJANGO_ALLOWED_HOSTS=$DOMAIN
DJANGO_CSRF_TRUSTED_ORIGINS=https://$DOMAIN
# PyTorch (CPU) in the image: training works on this server.
BRICKDNN_INSTALL_TORCH=true
# The database: without this line SQLite in the Docker volume. Supabase: docs/database.md
# DATABASE_URL=postgresql://…
EOF
  chown deploy:deploy "$APP/.env"
  chmod 600 "$APP/.env"
fi
DOMAIN="$(sed -n 's/^BRICKDNN_DOMAIN=//p' "$APP/.env")"

echo
echo "Done. The site will be at https://$DOMAIN after the first deploy."
echo "Host key for GitHub (repository variable DEPLOY_KNOWN_HOSTS):"
for key in /etc/ssh/ssh_host_ed25519_key.pub; do
  echo "  $(ip -4 route get 1.1.1.1 | sed -n 's/.* src \([0-9.]*\).*/\1/p') $(cut -d' ' -f1-2 "$key")"
done
