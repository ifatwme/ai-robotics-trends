# syntax=docker/dockerfile:1
# Static results website served by unprivileged nginx; content is mounted read-only at runtime.
FROM nginxinc/nginx-unprivileged:1.29-alpine
COPY docker/nginx/http.conf /etc/nginx/conf.d/default.conf
COPY docker/nginx/security-headers.inc /etc/nginx/conf.d/security-headers.inc
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
