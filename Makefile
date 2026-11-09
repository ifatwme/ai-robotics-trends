# Convenience targets — everything runs in Docker.
SHELL := /bin/bash
COMPOSE := docker compose

.PHONY: help build up down logs update analyse site cert install-timers clean
help:            ## show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'
build:           ## build pipeline + web images
	$(COMPOSE) --profile pipeline build
up:              ## start the web server
	$(COMPOSE) up -d web
down:            ## stop the web server
	$(COMPOSE) down
logs:            ## follow web logs
	$(COMPOSE) logs -f web
update:          ## collect fresh data + full analysis + render site
	mkdir -p data out site cache history && $(COMPOSE) run --rm pipeline collect all
analyse:         ## re-run analysis on existing data (no collection)
	$(COMPOSE) run --rm pipeline all
site:            ## re-render the site only
	$(COMPOSE) run --rm pipeline site
cert:            ## obtain/renew the TLS certificate (server)
	scripts/renew_cert.sh
install-timers:  ## install systemd timers (server, needs root)
	cp deploy/systemd/*.service deploy/systemd/*.timer /etc/systemd/system/ && systemctl daemon-reload
	systemctl enable --now aitrends-weekly.timer aitrends-cert-renew.timer
clean:           ## remove generated outputs (keeps data + caches)
	rm -rf out site
