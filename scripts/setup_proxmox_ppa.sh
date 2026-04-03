#!/usr/bin/env bash
# Add ppa:temonade-team/stable to Proxmox apt sources.
# Proxmox is Debian-based, so Launchpad PPAs must be added manually.
# Run as root on the Proxmox host.

set -euo pipefail

PPA_OWNER="temonade-team"
PPA_NAME="stable"
# Proxmox 8 is based on Debian Bookworm; use Ubuntu Jammy for PPA compatibility
UBUNTU_CODENAME="jammy"
KEYRING_FILE="/usr/share/keyrings/temonade-team-stable.gpg"
SOURCES_FILE="/etc/apt/sources.list.d/temonade-team-stable.list"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Error: this script must be run as root." >&2
    exit 1
fi

echo "→ Installing dependencies..."
apt-get install -y --no-install-recommends curl gpg

echo "→ Fetching GPG key for ppa:${PPA_OWNER}/${PPA_NAME}..."
# Launchpad PPA key fingerprint is retrieved via the keyserver
curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x$(
    curl -fsSL "https://launchpad.net/api/1.0/~${PPA_OWNER}/+archive/${PPA_NAME}" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['signing_key_fingerprint'])"
)" | gpg --dearmor -o "${KEYRING_FILE}"

echo "→ Adding apt source..."
cat > "${SOURCES_FILE}" <<EOF
deb [signed-by=${KEYRING_FILE}] https://ppa.launchpadcontent.net/${PPA_OWNER}/${PPA_NAME}/ubuntu ${UBUNTU_CODENAME} main
EOF

echo "→ Running apt-get update..."
apt-get update

echo ""
echo "Done! ppa:${PPA_OWNER}/${PPA_NAME} is now available."
echo "  Keyring : ${KEYRING_FILE}"
echo "  Sources : ${SOURCES_FILE}"
