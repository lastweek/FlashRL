#!/bin/bash
# Development environment setup and local metrics stack control for FlashRL

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$ROOT_DIR/metric/docker-compose.yml"
GRAFANA_URL="http://localhost:3000"
PROMETHEUS_URL="http://localhost:9090"
PUSHGATEWAY_URL="http://localhost:9091"
METRICS_READY_TIMEOUT_SECONDS="${FLASHRL_METRICS_READY_TIMEOUT_SECONDS:-60}"
METRICS_READY_INTERVAL_SECONDS="${FLASHRL_METRICS_READY_INTERVAL_SECONDS:-1}"

is_sourced() {
  [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

finish() {
  local status="$1"
  if is_sourced; then
    return "$status"
  fi
  exit "$status"
}

activate_env() {
  export PYTHONPYCACHEPREFIX="$ROOT_DIR/.cache/pycache"
  export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

  echo "✓ FlashRL dev environment activated"
  echo "  - Bytecode cache: .cache/pycache/"
  echo "  - PYTHONPATH set"
}

metrics_usage() {
  echo "Usage: ./dev.sh metrics <up|down|reset|status>"
}

require_command() {
  local command_name="$1"
  if command -v "$command_name" >/dev/null 2>&1; then
    return 0
  fi

  echo "Error: required command '$command_name' was not found in PATH." >&2
  return 1
}

compose() {
  docker compose -f "$COMPOSE_FILE" "$@"
}

service_health_url() {
  case "$1" in
    grafana)
      echo "$GRAFANA_URL/api/health"
      ;;
    prometheus)
      echo "$PROMETHEUS_URL/-/ready"
      ;;
    pushgateway)
      echo "$PUSHGATEWAY_URL/-/ready"
      ;;
    *)
      return 1
      ;;
  esac
}

service_label() {
  case "$1" in
    grafana)
      echo "Grafana"
      ;;
    prometheus)
      echo "Prometheus"
      ;;
    pushgateway)
      echo "Pushgateway"
      ;;
    *)
      return 1
      ;;
  esac
}

service_ready() {
  local service="$1"
  local url
  url="$(service_health_url "$service")" || return 1
  curl --fail --silent --show-error --output /dev/null "$url"
}

unready_services() {
  local services=""
  local service
  for service in grafana prometheus pushgateway; do
    if ! service_ready "$service"; then
      if [[ -n "$services" ]]; then
        services="$services $service"
      else
        services="$service"
      fi
    fi
  done
  printf '%s' "$services"
}

print_readiness_summary() {
  local service
  local status
  local url
  for service in grafana prometheus pushgateway; do
    url="$(service_health_url "$service")" || continue
    if service_ready "$service"; then
      status="ready"
    else
      status="not_ready"
    fi
    echo "  - $(service_label "$service"): $status ($url)"
  done
}

wait_for_metrics_stack() {
  local deadline=$((SECONDS + METRICS_READY_TIMEOUT_SECONDS))
  local unready=""

  while true; do
    unready="$(unready_services)"
    if [[ -z "$unready" ]]; then
      return 0
    fi

    if (( SECONDS >= deadline )); then
      echo "Error: metrics stack did not become ready within ${METRICS_READY_TIMEOUT_SECONDS}s." >&2
      echo "Still not ready: $unready" >&2
      return 1
    fi

    sleep "$METRICS_READY_INTERVAL_SECONDS"
  done
}

metrics_up() {
  require_command docker || return 1
  require_command curl || return 1

  compose up -d || return 1
  echo "Waiting for metrics endpoints to become ready..."
  if ! wait_for_metrics_stack; then
    echo "Current container status:" >&2
    compose ps >&2 || true
    return 1
  fi

  echo "✓ FlashRL metrics stack is running"
  echo "  - Grafana: $GRAFANA_URL"
  echo "  - Troubleshooting:"
  echo "    Prometheus: $PROMETHEUS_URL"
  echo "    Pushgateway: $PUSHGATEWAY_URL"
}

metrics_down() {
  require_command docker || return 1
  compose down || return 1
  echo "✓ FlashRL metrics stack stopped"
}

metrics_reset() {
  require_command docker || return 1
  compose down -v --remove-orphans || return 1
  echo "✓ FlashRL metrics stack reset"
}

metrics_status() {
  require_command docker || return 1
  require_command curl || return 1

  compose ps || return 1
  echo
  echo "Endpoint readiness:"
  print_readiness_summary
}

main() {
  if [[ $# -eq 0 ]]; then
    activate_env
    return 0
  fi

  case "$1" in
    metrics)
      local action="${2:-}"
      case "$action" in
        up)
          metrics_up
          ;;
        down)
          metrics_down
          ;;
        reset)
          metrics_reset
          ;;
        status)
          metrics_status
          ;;
        *)
          metrics_usage
          return 1
          ;;
      esac
      ;;
    help|-h|--help)
      echo "Usage:"
      echo "  source ./dev.sh"
      echo "  ./dev.sh metrics <up|down|reset|status>"
      ;;
    *)
      echo "Unknown command: $1" >&2
      echo
      echo "Usage:"
      echo "  source ./dev.sh"
      echo "  ./dev.sh metrics <up|down|reset|status>"
      return 1
      ;;
  esac
}

main "$@"
finish $?
