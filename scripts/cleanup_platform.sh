#!/usr/bin/env bash
set -euo pipefail

FLASHRL_JOB_RESOURCE="flashrljobs.platform.flashrl.dev"
FLASHRL_CRD_NAME="flashrljobs.platform.flashrl.dev"
POLL_SECONDS=5

OPERATOR_NAMESPACE="flashrl-system"
WAIT_SECONDS=300

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cleanup_platform.sh [--operator-namespace flashrl-system] [--wait-seconds 300]
EOF
}

log_section() {
  printf '\n== %s ==\n' "$1" >&2
}

log_info() {
  printf '[info] %s\n' "$*" >&2
}

log_cmd() {
  printf '[cmd]' >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
}

run_cmd() {
  log_cmd "$@"
  "$@"
}

capture_cmd() {
  log_cmd "$@"
  "$@"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --operator-namespace)
      OPERATOR_NAMESPACE=${2:-}
      shift 2
      ;;
    --wait-seconds)
      WAIT_SECONDS=${2:-}
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_command kubectl

log_section "Validate Cluster Connectivity"
run_cmd kubectl cluster-info >/dev/null
CURRENT_CONTEXT="$(capture_cmd kubectl config current-context)"
log_info "kubectl context: $CURRENT_CONTEXT"
log_info "Operator namespace: $OPERATOR_NAMESPACE"
log_info "Wait timeout: ${WAIT_SECONDS}s"

log_cmd kubectl get crd "$FLASHRL_CRD_NAME"
if kubectl get crd "$FLASHRL_CRD_NAME" >/dev/null 2>&1; then
  log_section "Delete FlashRLJob Resources"
  run_cmd kubectl delete "$FLASHRL_JOB_RESOURCE" --all --all-namespaces --ignore-not-found=true --wait=true

  deadline=$((SECONDS + WAIT_SECONDS))
  log_info "Waiting for all ${FLASHRL_JOB_RESOURCE} resources to finish deleting"
  while (( SECONDS < deadline )); do
    remaining_jobs="$(kubectl get "$FLASHRL_JOB_RESOURCE" --all-namespaces -o name 2>/dev/null || true)"
    if [[ -z "$remaining_jobs" ]]; then
      break
    fi
    remaining_count="$(printf '%s\n' "$remaining_jobs" | awk 'NF {count++} END {print count+0}')"
    log_info "Still waiting for ${remaining_count} FlashRLJob resource(s) to delete"
    sleep "$POLL_SECONDS"
  done

  if [[ -n "$(kubectl get "$FLASHRL_JOB_RESOURCE" --all-namespaces -o name 2>/dev/null || true)" ]]; then
    echo "Timed out waiting for FlashRLJob resources to delete." >&2
    exit 1
  fi
else
  log_info "CRD $FLASHRL_CRD_NAME is not installed; skipping FlashRLJob deletion"
fi

log_section "Delete FlashRL Operator Deployment"
run_cmd kubectl delete deployment flashrl-operator -n "$OPERATOR_NAMESPACE" --ignore-not-found=true --wait=true

log_section "Delete FlashRL Cluster RBAC"
run_cmd kubectl delete clusterrolebinding flashrl-operator --ignore-not-found=true
run_cmd kubectl delete clusterrole flashrl-operator --ignore-not-found=true

log_section "Delete FlashRL Operator Namespace"
run_cmd kubectl delete namespace "$OPERATOR_NAMESPACE" --ignore-not-found=true --wait=true

log_section "Delete FlashRL CRD"
run_cmd kubectl delete crd "$FLASHRL_CRD_NAME" --ignore-not-found=true --wait=true

echo "FlashRL platform cleanup complete."
