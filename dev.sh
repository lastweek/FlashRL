#!/bin/bash
# Development environment setup and local metrics stack control for FlashRL

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$ROOT_DIR/metric/docker-compose.yml"
GRAFANA_URL="http://localhost:3000"
PROMETHEUS_URL="http://localhost:9090"
PUSHGATEWAY_URL="http://localhost:9091"
DEFAULT_VLLM_VENV="$HOME/.venv-vllm"
DEFAULT_VLLM_METAL_VENV="$HOME/.venv-vllm-metal"
METRICS_READY_TIMEOUT_SECONDS="${FLASHRL_METRICS_READY_TIMEOUT_SECONDS:-60}"
METRICS_READY_INTERVAL_SECONDS="${FLASHRL_METRICS_READY_INTERVAL_SECONDS:-1}"

is_sourced() {
  [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

finish() {
  local exit_status="$1"
  if is_sourced; then
    return "$exit_status"
  fi
  exit "$exit_status"
}

activate_env() {
  export PYTHONPYCACHEPREFIX="$ROOT_DIR/.cache/pycache"
  export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
  auto_export_vllm_python

  echo "✓ FlashRL dev environment activated"
  echo "  - Bytecode cache: .cache/pycache/"
  echo "  - PYTHONPATH set"
  if [[ -n "${FLASHRL_VLLM_PYTHON:-}" ]]; then
    echo "  - FLASHRL_VLLM_PYTHON: $FLASHRL_VLLM_PYTHON"
  fi
}

metrics_usage() {
  echo "Usage: ./dev.sh metrics <up|down|reset|status>"
}

vllm_usage() {
  echo "Usage: ./dev.sh vllm <setup|status>"
}

require_command() {
  local command_name="$1"
  if command -v "$command_name" >/dev/null 2>&1; then
    return 0
  fi

  echo "Error: required command '$command_name' was not found in PATH." >&2
  return 1
}

is_macos_arm64() {
  [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]
}

default_vllm_python() {
  if is_macos_arm64 && [[ -x "$DEFAULT_VLLM_METAL_VENV/bin/python" ]]; then
    printf '%s\n' "$DEFAULT_VLLM_METAL_VENV/bin/python"
    return 0
  fi
  if [[ -x "$DEFAULT_VLLM_VENV/bin/python" ]]; then
    printf '%s\n' "$DEFAULT_VLLM_VENV/bin/python"
    return 0
  fi
  return 1
}

auto_export_vllm_python() {
  if [[ -n "${FLASHRL_VLLM_PYTHON:-}" ]]; then
    return 0
  fi

  local runtime_python=""
  if runtime_python="$(default_vllm_python)"; then
    export FLASHRL_VLLM_PYTHON="$runtime_python"
  fi
}

print_vllm_export_hint() {
  local python_path="$1"
  echo "export FLASHRL_VLLM_PYTHON=\"$python_path\""
}

python_has_module() {
  local python_path="$1"
  local module_name="$2"
  "$python_path" -c "import $module_name" >/dev/null 2>&1
}

linux_vllm_python_command() {
  local candidate=""
  for candidate in python3.13 python3.12 python3; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    if "$candidate" -c 'import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)'; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

vllm_status() {
  local runtime_python=""
  local source_label="prepared runtime"

  if [[ -n "${FLASHRL_VLLM_PYTHON:-}" ]]; then
    runtime_python="$FLASHRL_VLLM_PYTHON"
    source_label="FLASHRL_VLLM_PYTHON"
  elif runtime_python="$(default_vllm_python)"; then
    source_label="default prepared runtime"
  else
    echo "Prepared vLLM runtime: not found"
    if is_macos_arm64; then
      echo "Run: ./dev.sh vllm setup"
    else
      echo "Run: ./dev.sh vllm setup"
      echo "Or install FlashRL with the optional vllm extra in your current environment."
    fi
    return 0
  fi

  echo "Prepared vLLM runtime: $runtime_python"
  echo "Source: $source_label"
  print_vllm_export_hint "$runtime_python"
}

vllm_setup_macos() {
  require_command curl || return 1
  require_command uv || return 1

  if [[ -x "$DEFAULT_VLLM_METAL_VENV/bin/python" ]] && python_has_module "$DEFAULT_VLLM_METAL_VENV/bin/python" vllm_metal; then
    echo "✓ Existing vllm-metal runtime found at $DEFAULT_VLLM_METAL_VENV"
  else
    echo "Installing vllm-metal into $DEFAULT_VLLM_METAL_VENV ..."
    if ! curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash; then
      echo "Installer did not complete cleanly; falling back to source install for vllm-metal ..."
    fi
  fi

  if [[ ! -x "$DEFAULT_VLLM_METAL_VENV/bin/python" ]]; then
    echo "Error: vllm-metal setup did not create $DEFAULT_VLLM_METAL_VENV/bin/python" >&2
    return 1
  fi

  if ! python_has_module "$DEFAULT_VLLM_METAL_VENV/bin/python" vllm_metal; then
    require_command git || return 1
    echo "Installing vllm-metal from source fallback ..."
    uv pip install --python "$DEFAULT_VLLM_METAL_VENV/bin/python" \
      git+https://github.com/vllm-project/vllm-metal.git || return 1
  fi

  if ! python_has_module "$DEFAULT_VLLM_METAL_VENV/bin/python" vllm_metal; then
    echo "Error: vllm-metal setup completed but the runtime still cannot import vllm_metal." >&2
    return 1
  fi

  echo
  echo "Prepared vLLM runtime:"
  print_vllm_export_hint "$DEFAULT_VLLM_METAL_VENV/bin/python"
}

vllm_setup_linux() {
  require_command uv || return 1

  local python_cmd=""
  if ! python_cmd="$(linux_vllm_python_command)"; then
    echo "Error: Linux vLLM setup requires python3.12 or python3.13 in PATH." >&2
    return 1
  fi

  echo "Preparing vLLM runtime with $python_cmd ..."
  uv venv --python "$python_cmd" "$DEFAULT_VLLM_VENV" || return 1
  uv pip install --python "$DEFAULT_VLLM_VENV/bin/python" "vllm>=0.16.0" || return 1

  echo
  echo "Prepared vLLM runtime:"
  print_vllm_export_hint "$DEFAULT_VLLM_VENV/bin/python"
}

vllm_setup() {
  if is_macos_arm64; then
    vllm_setup_macos
    return $?
  fi
  if [[ "$(uname -s)" == "Linux" ]]; then
    vllm_setup_linux
    return $?
  fi

  echo "Error: unsupported platform for automatic vLLM setup: $(uname -s) $(uname -m)" >&2
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
  local readiness
  local url
  for service in grafana prometheus pushgateway; do
    url="$(service_health_url "$service")" || continue
    if service_ready "$service"; then
      readiness="ready"
    else
      readiness="not_ready"
    fi
    echo "  - $(service_label "$service"): $readiness ($url)"
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
    vllm)
      local action="${2:-}"
      case "$action" in
        setup)
          vllm_setup
          ;;
        status)
          vllm_status
          ;;
        *)
          vllm_usage
          return 1
          ;;
      esac
      ;;
    help|-h|--help)
      echo "Usage:"
      echo "  source ./dev.sh"
      echo "  ./dev.sh metrics <up|down|reset|status>"
      echo "  ./dev.sh vllm <setup|status>"
      ;;
    *)
      echo "Unknown command: $1" >&2
      echo
      echo "Usage:"
      echo "  source ./dev.sh"
      echo "  ./dev.sh metrics <up|down|reset|status>"
      echo "  ./dev.sh vllm <setup|status>"
      return 1
      ;;
  esac
}

if is_sourced; then
  main
  finish $?
else
  main "$@"
  finish $?
fi
