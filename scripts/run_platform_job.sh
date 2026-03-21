#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FLASHRL_JOB_RESOURCE="flashrljobs.platform.flashrl.dev"
FLASHRL_CRD_NAME="flashrljobs.platform.flashrl.dev"
POLL_SECONDS=5

CONFIG_PATH=""
IMAGE_ENV_FILE=""
TIMEOUT_SECONDS=1800
OPERATOR_NAMESPACE="flashrl-system"
ARTIFACT_DIR=""
DEFAULT_ARTIFACT_TAG="run-$(date +%Y%m%d-%H%M%S)"

TMP_DIR="$(mktemp -d)"
JOB_NAME=""
JOB_NAMESPACE=""
CHECKPOINT_INSPECT_PATH=""
WEIGHT_INSPECT_PATH=""
VERIFY_ARTIFACTS=0
CONTROLLER_POD=""
JOB_LOG_ROOT=""
ACTIVE_CONTROLLER_RUN_DIR=""
ANNOUNCED_JOB_LOG_ROOT=""
ANNOUNCED_CONTROLLER_RUN_DIR=""
STATUS_EVENTS_FILE="$TMP_DIR/status-events.seen"

# Color support detection
if [[ -t 2 ]] && [[ -z "${NO_COLOR:-}" ]]; then
  # Terminal supports colors and NO_COLOR is not set
  COLOR_RESET='\033[0m'
  COLOR_BOLD='\033[1m'
  COLOR_RED='\033[1;31m'
  COLOR_GREEN='\033[1;32m'
  COLOR_YELLOW='\033[1;33m'
  COLOR_BLUE='\033[1;34m'
  COLOR_MAGENTA='\033[1;35m'
  COLOR_CYAN='\033[1;36m'
  COLOR_WHITE='\033[1;37m'
else
  # Fallback to no colors
  COLOR_RESET=''
  COLOR_BOLD=''
  COLOR_RED=''
  COLOR_GREEN=''
  COLOR_YELLOW=''
  COLOR_BLUE=''
  COLOR_MAGENTA=''
  COLOR_CYAN=''
  COLOR_WHITE=''
fi

# Script start time for elapsed time tracking
SCRIPT_START_TIME="${SECONDS}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_platform_job.sh \
    --config <path> \
    --image-env-file <path> \
    [--timeout-seconds 1800] \
    [--operator-namespace flashrl-system] \
    [--artifact-dir logs/platform-runs/...]
EOF
}

log_section() {
  local title="$1"
  local line="═══════════════════════════════════════════════════════════════"
  printf '\n${COLOR_MAGENTA}%s\n' "$line" >&2
  printf '  %s\n' "$title" >&2
  printf '%s${COLOR_RESET}\n\n' "$line" >&2
}

log_info() {
  printf '${COLOR_BLUE}[info]${COLOR_RESET} %s\n' "$*" >&2
}

log_success() {
  printf '${COLOR_GREEN}✓${COLOR_RESET} %s\n' "$*" >&2
}

log_warn() {
  printf '${COLOR_YELLOW}⚠${COLOR_RESET} %s\n' "$*" >&2
}

log_error() {
  printf '${COLOR_RED}✗${COLOR_RESET} %s\n' "$*" >&2
}

log_cmd() {
  printf '${COLOR_CYAN}[cmd]${COLOR_RESET}' >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
}

log_cmd_text() {
  printf '${COLOR_CYAN}[cmd]${COLOR_RESET} %s\n' "$1" >&2
}

run_cmd() {
  log_cmd "$@"
  "$@"
}

capture_cmd() {
  log_cmd "$@"
  "$@"
}

run_labeled_cmd() {
  local label=$1
  shift
  log_cmd_text "$label"
  "$@"
}

format_elapsed_time() {
  local total_seconds=$1
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))

  if ((hours > 0)); then
    printf "%dh %dm %ds" "$hours" "$minutes" "$seconds"
  elif ((minutes > 0)); then
    printf "%dm %ds" "$minutes" "$seconds"
  else
    printf "%ds" "$seconds"
  fi
}

run_cmd_stdout_to_file() {
  local output_path=$1
  shift
  log_info "Writing command stdout to $output_path"
  log_cmd "$@"
  "$@" > "$output_path"
}

run_cmd_all_to_file() {
  local output_path=$1
  shift
  log_info "Writing command output to $output_path"
  log_cmd "$@"
  "$@" > "$output_path" 2>&1
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log_error "Required command not found: $1"
    exit 1
  fi
}

validate_local_image_mode() {
  local current_context=$1
  local cluster_type=${FLASHRL_LOCAL_CLUSTER_TYPE:-}

  if [[ "${FLASHRL_IMAGE_MODE:-registry}" != "local" ]]; then
    return
  fi

  case "$cluster_type" in
    minikube)
      local expected_context="${FLASHRL_LOCAL_CLUSTER_PROFILE:-minikube}"
      if [[ "$current_context" != "$expected_context" ]]; then
        log_error "Local image env file targets minikube profile ${expected_context}, but current kubectl context is ${current_context}"
        exit 1
      fi
      ;;
    kind)
      local expected_name="${FLASHRL_LOCAL_CLUSTER_NAME:-kind}"
      local expected_context="kind-${expected_name}"
      if [[ "$current_context" != "$expected_context" ]]; then
        log_error "Local image env file targets kind cluster ${expected_name}, but current kubectl context is ${current_context}"
        exit 1
      fi
      ;;
    docker-desktop)
      if [[ "$current_context" != "docker-desktop" ]]; then
        log_error "Local image env file targets docker-desktop, but current kubectl context is ${current_context}"
        exit 1
      fi
      ;;
    *)
      log_error "Unsupported or missing FLASHRL_LOCAL_CLUSTER_TYPE for local image mode: ${cluster_type:-<empty>}"
      exit 1
      ;;
  esac
}

cleanup_tmp() {
  rm -rf "$TMP_DIR"
}

collect_diagnostics() {
  if [[ -z "$ARTIFACT_DIR" ]]; then
    return
  fi

  set +e
  mkdir -p "$ARTIFACT_DIR"

  run_cmd_all_to_file "$ARTIFACT_DIR/kube-context.txt" kubectl config current-context
  run_cmd_all_to_file "$ARTIFACT_DIR/namespaces.txt" kubectl get namespaces
  run_cmd_all_to_file "$ARTIFACT_DIR/crds.txt" kubectl get crd
  run_cmd_all_to_file "$ARTIFACT_DIR/kubectl-get-all.txt" kubectl get pods,deployments,statefulsets,services,pvc --all-namespaces -o wide
  run_cmd_all_to_file "$ARTIFACT_DIR/operator.log" kubectl logs -n "$OPERATOR_NAMESPACE" -l app.kubernetes.io/name=flashrl-operator --tail=500 --prefix=true

  if [[ -n "$JOB_NAME" && -n "$JOB_NAMESPACE" ]]; then
    run_cmd_all_to_file "$ARTIFACT_DIR/flashrljob.yaml" kubectl get "$FLASHRL_JOB_RESOURCE" "$JOB_NAME" -n "$JOB_NAMESPACE" -o yaml
    run_cmd_all_to_file "$ARTIFACT_DIR/flashrljob-describe.txt" kubectl describe "$FLASHRL_JOB_RESOURCE" "$JOB_NAME" -n "$JOB_NAMESPACE"
    run_cmd_all_to_file "$ARTIFACT_DIR/pods-describe.txt" kubectl describe pods -n "$JOB_NAMESPACE" -l "flashrl.dev/job=$JOB_NAME"

    for component in controller learner serving rollout reward; do
      run_cmd_all_to_file "$ARTIFACT_DIR/${component}.log" \
        kubectl logs \
        -n "$JOB_NAMESPACE" \
        -l "flashrl.dev/job=$JOB_NAME,app.kubernetes.io/component=$component" \
        --tail=500 \
        --prefix=true
    done
  fi
  set -e
}

on_error() {
  local exit_code=$1
  local line_no=$2
  trap - ERR
  printf '${COLOR_RED}✗ run_platform_job.sh failed at line %d with exit code %d${COLOR_RESET}\n' "$line_no" "$exit_code" >&2
  if [[ -n "$ARTIFACT_DIR" ]]; then
    printf '${COLOR_YELLOW}⚠ Writing diagnostics to %s${COLOR_RESET}\n' "$ARTIFACT_DIR" >&2
  fi
  collect_diagnostics
  cleanup_tmp
  exit "$exit_code"
}

wait_for_workload_rollout() {
  local kind=$1
  local name=$2
  local namespace=$3
  local deadline=$((SECONDS + TIMEOUT_SECONDS))

  log_info "Waiting for ${kind}/${name} in namespace=${namespace}"
  while (( SECONDS < deadline )); do
    if kubectl get "$kind" "$name" -n "$namespace" >/dev/null 2>&1; then
      run_cmd kubectl rollout status "${kind}/${name}" -n "$namespace" --timeout="${TIMEOUT_SECONDS}s"
      return
    fi
    sleep "$POLL_SECONDS"
  done

  log_error "Timed out waiting for ${kind}/${name} in namespace=${namespace}"
  return 1
}

wait_for_job_completion() {
  local deadline=$((SECONDS + TIMEOUT_SECONDS))
  local status_json=""
  local job_phase=""
  local finished_at=""
  local last_completed_step=""
  local last_error=""
  local job_log_root=""
  local active_controller_run_dir=""
  local new_events=""

  log_info "Polling ${FLASHRL_JOB_RESOURCE}/${JOB_NAME} in namespace=${JOB_NAMESPACE} until finishedAt is set"
  while (( SECONDS < deadline )); do
    status_json="$(kubectl get "$FLASHRL_JOB_RESOURCE" "$JOB_NAME" -n "$JOB_NAMESPACE" -o json)"
    eval "$(
      python3 - <<'PY' <<<"$status_json"
import json
import shlex
import sys

payload = json.load(sys.stdin)
status = payload.get("status") or {}
progress = status.get("progress") or {}
fields = {
    "JOB_PHASE": status.get("phase") or "",
    "JOB_FINISHED_AT": status.get("finishedAt") or "",
    "JOB_LAST_COMPLETED_STEP": progress.get("lastCompletedStep", 0) or 0,
    "JOB_LAST_ERROR": status.get("lastError") or "",
    "JOB_LOG_ROOT": status.get("logRoot") or "",
    "ACTIVE_CONTROLLER_RUN_DIR": status.get("activeControllerRunDir") or "",
}
for key, value in fields.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
    )"

    job_phase="$JOB_PHASE"
    finished_at="$JOB_FINISHED_AT"
    last_completed_step="$JOB_LAST_COMPLETED_STEP"
    last_error="$JOB_LAST_ERROR"
    job_log_root="$JOB_LOG_ROOT"
    active_controller_run_dir="$ACTIVE_CONTROLLER_RUN_DIR"

    if [[ -n "$job_log_root" && "$ANNOUNCED_JOB_LOG_ROOT" != "$job_log_root" ]]; then
      ANNOUNCED_JOB_LOG_ROOT="$job_log_root"
    fi
    if [[ -n "$active_controller_run_dir" && "$ANNOUNCED_CONTROLLER_RUN_DIR" != "$active_controller_run_dir" ]]; then
      ANNOUNCED_CONTROLLER_RUN_DIR="$active_controller_run_dir"
    fi

    printf '\r${COLOR_BLUE}ℹ${COLOR_RESET} phase=${COLOR_BOLD}%s${COLOR_RESET} lastCompletedStep=${COLOR_BOLD}%s${COLOR_RESET} finishedAt=${COLOR_BOLD}%s${COLOR_RESET}' \
      "${job_phase:-<none>}" "${last_completed_step:-0}" "${finished_at:-<none>}" >&2
    if [[ -n "$last_error" ]]; then
      printf ' ${COLOR_RED}lastError=%s${COLOR_RESET}' "$last_error" >&2
    fi
    printf '\r' >&2

    if [[ -n "$ANNOUNCED_JOB_LOG_ROOT" ]]; then
      log_info "Job log root: $ANNOUNCED_JOB_LOG_ROOT"
      ANNOUNCED_JOB_LOG_ROOT=""
    fi
    if [[ -n "$ANNOUNCED_CONTROLLER_RUN_DIR" ]]; then
      log_info "Controller run dir: $ANNOUNCED_CONTROLLER_RUN_DIR"
      ANNOUNCED_CONTROLLER_RUN_DIR=""
    fi

    new_events="$(python3 - "$STATUS_EVENTS_FILE" <<'PY' <<<"$status_json"
import json
from pathlib import Path
import sys

seen_path = Path(sys.argv[1])
payload = json.load(sys.stdin)
status = payload.get("status") or {}
events = status.get("events") or []
seen = set()
if seen_path.exists():
    seen = {
        line.strip()
        for line in seen_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
new_lines = []
new_keys = []
for item in events:
    if not isinstance(item, dict):
        continue
    key = "|".join(
        str(item.get(field, ""))
        for field in ("timestamp", "component", "event", "message")
    )
    if key in seen:
        continue
    new_keys.append(key)
    component = item.get("component") or "job"
    new_lines.append(
        f"{item.get('timestamp', '<unknown>')} [{component}] {item.get('event', 'event')} {item.get('message', '')}"
    )
if new_keys:
    with seen_path.open("a", encoding="utf-8") as handle:
        for key in new_keys:
            handle.write(key + "\n")
print("\n".join(new_lines))
PY
    )"
    if [[ -n "$new_events" ]]; then
      printf '\n' >&2
      while IFS= read -r event_line; do
        [[ -n "$event_line" ]] || continue
        log_info "$event_line"
      done <<<"$new_events"
    fi

    if [[ "$job_phase" == "Failed" ]]; then
      printf '\n' >&2
      log_error "FlashRLJob $JOB_NAME failed"
      return 1
    fi
    if [[ -n "$finished_at" ]]; then
      printf '\n' >&2
      return 0
    fi

    sleep "$POLL_SECONDS"
  done

  printf '\n' >&2
  log_error "Timed out waiting for FlashRLJob $JOB_NAME to finish"
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH=${2:-}
      shift 2
      ;;
    --image-env-file)
      IMAGE_ENV_FILE=${2:-}
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS=${2:-}
      shift 2
      ;;
    --operator-namespace)
      OPERATOR_NAMESPACE=${2:-}
      shift 2
      ;;
    --artifact-dir)
      ARTIFACT_DIR=${2:-}
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

if [[ -z "$CONFIG_PATH" || -z "$IMAGE_ENV_FILE" ]]; then
  log_error "Missing required arguments"
  usage >&2
  exit 1
fi

if [[ -z "$ARTIFACT_DIR" ]]; then
  ARTIFACT_DIR="$REPO_ROOT/logs/platform-runs/$DEFAULT_ARTIFACT_TAG"
fi

require_command kubectl
require_command python3

if [[ ! -f "$IMAGE_ENV_FILE" ]]; then
  log_error "Image env file not found: $IMAGE_ENV_FILE"
  exit 1
fi

trap 'on_error $? $LINENO' ERR
trap cleanup_tmp EXIT

log_section "Validate Cluster Connectivity"
run_cmd kubectl cluster-info >/dev/null
CURRENT_CONTEXT="$(capture_cmd kubectl config current-context)"
log_success "kubectl context: $CURRENT_CONTEXT"

log_section "Load Image References"
log_info "Loading image environment from $IMAGE_ENV_FILE"
set -a
# shellcheck disable=SC1090
source "$IMAGE_ENV_FILE"
set +a

: "${FLASHRL_OPERATOR_IMAGE:?FLASHRL_OPERATOR_IMAGE is required in the image env file}"
: "${FLASHRL_RUNTIME_IMAGE:?FLASHRL_RUNTIME_IMAGE is required in the image env file}"
: "${FLASHRL_SERVING_IMAGE:?FLASHRL_SERVING_IMAGE is required in the image env file}"
: "${FLASHRL_TRAINING_IMAGE:?FLASHRL_TRAINING_IMAGE is required in the image env file}"

log_success "Image references loaded successfully"

validate_local_image_mode "$CURRENT_CONTEXT"
log_info "Config path: $CONFIG_PATH"
log_info "Artifact dir: $ARTIFACT_DIR"
log_info "Image mode: ${FLASHRL_IMAGE_MODE:-registry}"
log_info "Operator image: $FLASHRL_OPERATOR_IMAGE"
log_info "Runtime image: $FLASHRL_RUNTIME_IMAGE"
log_info "Serving image: $FLASHRL_SERVING_IMAGE"
log_info "Training image: $FLASHRL_TRAINING_IMAGE"

PATCHED_NAMESPACE_YAML="$TMP_DIR/namespace.yaml"
PATCHED_RBAC_YAML="$TMP_DIR/operator-rbac.yaml"
PATCHED_OPERATOR_YAML="$TMP_DIR/operator.yaml"
RENDERED_JOB_YAML="$TMP_DIR/rendered-job.yaml"
PATCHED_JOB_YAML="$TMP_DIR/job.yaml"
PATCHED_JOB_ENV="$TMP_DIR/job.env"

log_section "Prepare Operator Manifests"
run_labeled_cmd "python3 - <patch namespace manifest>" python3 - "$REPO_ROOT/flashrl/platform/k8s/namespace.yaml" "$OPERATOR_NAMESPACE" "$PATCHED_NAMESPACE_YAML" <<'PY'
from pathlib import Path
import sys
import yaml

src, namespace, dst = sys.argv[1:4]
payload = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
payload["metadata"]["name"] = namespace
Path(dst).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY

run_labeled_cmd "python3 - <patch operator RBAC manifest>" python3 - "$REPO_ROOT/flashrl/platform/k8s/operator-rbac.yaml" "$OPERATOR_NAMESPACE" "$PATCHED_RBAC_YAML" <<'PY'
from pathlib import Path
import sys
import yaml

src, namespace, dst = sys.argv[1:4]
docs = [item for item in yaml.safe_load_all(Path(src).read_text(encoding="utf-8")) if item is not None]
for doc in docs:
    if doc.get("kind") == "ServiceAccount":
        doc["metadata"]["namespace"] = namespace
    if doc.get("kind") == "ClusterRoleBinding":
        for subject in doc.get("subjects", []):
            if subject.get("kind") == "ServiceAccount" and subject.get("name") == "flashrl-operator":
                subject["namespace"] = namespace
Path(dst).write_text(yaml.safe_dump_all(docs, sort_keys=False), encoding="utf-8")
PY

run_labeled_cmd "python3 - <patch operator manifest>" python3 - "$REPO_ROOT/flashrl/platform/k8s/operator.yaml" "$OPERATOR_NAMESPACE" "$FLASHRL_OPERATOR_IMAGE" "$PATCHED_OPERATOR_YAML" <<'PY'
from pathlib import Path
import sys
import yaml

src, namespace, image, dst = sys.argv[1:5]
payload = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
payload["metadata"]["namespace"] = namespace
payload["spec"]["template"]["spec"]["containers"][0]["image"] = image
Path(dst).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY

log_section "Install FlashRL Operator"
run_cmd kubectl apply -f "$PATCHED_NAMESPACE_YAML"
run_cmd kubectl apply -f "$REPO_ROOT/flashrl/platform/k8s/job-crd.yaml"
run_cmd kubectl apply -f "$PATCHED_RBAC_YAML"
run_cmd kubectl apply -f "$PATCHED_OPERATOR_YAML"
run_cmd kubectl rollout status deployment/flashrl-operator -n "$OPERATOR_NAMESPACE" --timeout="${TIMEOUT_SECONDS}s"
log_success "FlashRL Operator installed successfully"

log_section "Render FlashRLJob"
run_cmd python3 -m flashrl platform render \
  --config "$CONFIG_PATH" \
  --output "$RENDERED_JOB_YAML"

run_labeled_cmd "python3 - <patch rendered FlashRLJob and derive inspect paths>" python3 - "$RENDERED_JOB_YAML" "$PATCHED_JOB_YAML" "$PATCHED_JOB_ENV" "$FLASHRL_RUNTIME_IMAGE" "$FLASHRL_SERVING_IMAGE" "$FLASHRL_TRAINING_IMAGE" <<'PY'
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
import shlex
import sys
import yaml

src, dst, env_path, runtime_image, serving_image, training_image = sys.argv[1:7]
job = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
spec = job["spec"]
images = spec["images"]
images["runtime"] = runtime_image
images["serving"] = serving_image
images["training"] = training_image

metadata = job.setdefault("metadata", {})
job_name = str(metadata["name"])
job_namespace = str(metadata.get("namespace") or "default")
metadata["namespace"] = job_namespace

shared_storage = spec.get("sharedStorage") or {}
checkpoint_path = ""
weight_path = ""

if bool(shared_storage.get("enabled", False)):
    mount_path = PurePosixPath(str(shared_storage["mountPath"]))
    checkpoint_path = str(mount_path / str(shared_storage.get("checkpointsSubPath", "checkpoints")))
    weight_path = str(mount_path / str(shared_storage.get("weightsSubPath", "weights")))
else:
    def local_path(raw: str) -> str:
        parsed = urlparse(str(raw))
        if parsed.scheme == "file":
            return parsed.path
        if parsed.scheme == "":
            return str(raw)
        return ""

    checkpoint_path = local_path(str(spec["storage"]["checkpoints"]["uriPrefix"]))
    weight_path = local_path(str(spec["storage"]["weights"]["uriPrefix"]))

verify_artifacts = "1" if checkpoint_path and weight_path else "0"

Path(dst).write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
with open(env_path, "w", encoding="utf-8") as handle:
    values = {
        "JOB_NAME": job_name,
        "JOB_NAMESPACE": job_namespace,
        "CHECKPOINT_INSPECT_PATH": checkpoint_path,
        "WEIGHT_INSPECT_PATH": weight_path,
        "VERIFY_ARTIFACTS": verify_artifacts,
    }
    for key, value in values.items():
        handle.write(f"{key}={shlex.quote(str(value))}\n")
PY

set -a
# shellcheck disable=SC1090
source "$PATCHED_JOB_ENV"
set +a

mkdir -p "$ARTIFACT_DIR"
log_info "Resolved job name: $JOB_NAME"
log_info "Resolved job namespace: $JOB_NAMESPACE"
if [[ "$VERIFY_ARTIFACTS" == "1" ]]; then
  log_info "Checkpoint inspect path: $CHECKPOINT_INSPECT_PATH"
  log_info "Weight inspect path: $WEIGHT_INSPECT_PATH"
else
  log_info "Artifact verification: skipped because storage paths are not inspectable filesystem paths"
fi

log_section "Ensure Job Namespace"
log_cmd kubectl get namespace "$JOB_NAMESPACE"
if kubectl get namespace "$JOB_NAMESPACE" >/dev/null 2>&1; then
  log_info "Namespace $JOB_NAMESPACE already exists"
else
  run_cmd kubectl create namespace "$JOB_NAMESPACE"
  log_success "Namespace $JOB_NAMESPACE created"
fi

log_section "Delete Previous FlashRLJob"
run_cmd kubectl delete "$FLASHRL_JOB_RESOURCE" "$JOB_NAME" -n "$JOB_NAMESPACE" --ignore-not-found=true --wait=true

log_section "Apply FlashRLJob"
run_cmd kubectl apply -f "$PATCHED_JOB_YAML"
log_success "FlashRLJob $JOB_NAME applied successfully"

log_section "Wait for FlashRL Workloads"
wait_for_workload_rollout deployment "${JOB_NAME}-controller" "$JOB_NAMESPACE"
wait_for_workload_rollout statefulset "${JOB_NAME}-learner" "$JOB_NAMESPACE"
wait_for_workload_rollout deployment "${JOB_NAME}-serving" "$JOB_NAMESPACE"
wait_for_workload_rollout deployment "${JOB_NAME}-rollout" "$JOB_NAMESPACE"
wait_for_workload_rollout deployment "${JOB_NAME}-reward" "$JOB_NAMESPACE"
log_success "All FlashRL workloads rolled out successfully"

log_section "Wait for FlashRLJob Completion"
wait_for_job_completion

log_section "Verify Artifacts"
CONTROLLER_POD="$(capture_cmd kubectl get pods -n "$JOB_NAMESPACE" -l "flashrl.dev/job=$JOB_NAME,app.kubernetes.io/component=controller" -o jsonpath='{.items[0].metadata.name}')"
if [[ -z "$CONTROLLER_POD" ]]; then
  log_error "Controller pod not found for $JOB_NAME"
  exit 1
fi
log_success "Controller pod: $CONTROLLER_POD"

if [[ "$VERIFY_ARTIFACTS" == "1" ]]; then
  run_cmd_stdout_to_file "$ARTIFACT_DIR/checkpoints.txt" kubectl exec -n "$JOB_NAMESPACE" "$CONTROLLER_POD" -- sh -c 'find "$1" -type f | sort' sh "$CHECKPOINT_INSPECT_PATH"
  run_cmd_stdout_to_file "$ARTIFACT_DIR/weights.txt" kubectl exec -n "$JOB_NAMESPACE" "$CONTROLLER_POD" -- sh -c 'find "$1" -type f | sort' sh "$WEIGHT_INSPECT_PATH"

  if [[ ! -s "$ARTIFACT_DIR/checkpoints.txt" ]]; then
    log_error "No checkpoint artifacts found under $CHECKPOINT_INSPECT_PATH"
    exit 1
  fi
  if [[ ! -s "$ARTIFACT_DIR/weights.txt" ]]; then
    log_error "No weight artifacts found under $WEIGHT_INSPECT_PATH"
    exit 1
  fi
  log_success "Artifact verification completed"
else
  log_warn "Skipping artifact file verification because storage paths are not inspectable filesystem paths"
fi

trap - ERR

log_section "FlashRLJob Finished"
log_success "Job: $JOB_NAME"
log_success "Namespace: $JOB_NAMESPACE"
log_success "Artifacts: $ARTIFACT_DIR"
ELAPSED_TIME=$((SECONDS - SCRIPT_START_TIME))
log_info "Total execution time: $(format_elapsed_time "$ELAPSED_TIME")"
echo
echo "${COLOR_GREEN}Inspect:${COLOR_RESET}"
echo "  kubectl get $FLASHRL_JOB_RESOURCE $JOB_NAME -n $JOB_NAMESPACE -o yaml"
echo "  kubectl get pods -n $JOB_NAMESPACE -l flashrl.dev/job=$JOB_NAME"
echo "  kubectl logs -n $JOB_NAMESPACE -l flashrl.dev/job=$JOB_NAME --prefix=true"
echo
echo "${COLOR_YELLOW}Cleanup:${COLOR_RESET}"
echo "  bash scripts/cleanup_platform.sh --operator-namespace $OPERATOR_NAMESPACE"
