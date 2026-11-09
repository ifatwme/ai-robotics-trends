#!/usr/bin/env bash
# Obtain/renew a Let's Encrypt certificate (short-lived IP-address profile by default) and hot-reload nginx.
# Usage: CERT_IDENTIFIER=<ip-or-domain> scripts/renew_cert.sh      (reads .env if present)
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && set -a && . ./.env && set +a
ID="${CERT_IDENTIFIER:?set CERT_IDENTIFIER (server IP or domain) in .env}"
PROFILE="${CERT_PROFILE:-shortlived}"
mkdir -p deploy/certs deploy/acme deploy/tls
# lego v5: `run` obtains the certificate or renews it when due (short-lived certs: at 1/2 of their lifetime).
docker run --rm -v "$PWD/deploy/certs:/certs" -v "$PWD/deploy/acme:/webroot" goacme/lego:latest run \
  --accept-tos --server https://acme-v02.api.letsencrypt.org/directory --domains "$ID" \
  --http --http.webroot /webroot --path /certs --profile "$PROFILE" ${LEGO_EXTRA:-}
install -o 101 -g 101 -m 0444 "deploy/certs/certificates/$ID.crt" deploy/tls/fullchain.pem   # 101 = nginx user
install -o 101 -g 101 -m 0400 "deploy/certs/certificates/$ID.key" deploy/tls/privkey.pem
chmod 755 deploy/tls
if docker ps --format '{{.Names}}' | grep -qx aitrends-web; then docker exec aitrends-web nginx -s reload; fi
echo "[$(date -u +%FT%TZ)] certificate valid until: $(openssl x509 -in deploy/tls/fullchain.pem -noout -enddate)"
