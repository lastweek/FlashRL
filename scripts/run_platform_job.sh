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

log_cmd_text() {
  printf '[cmd] %s\n' "$1" >&2
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
    echo "Required command not found: $1" >&2
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
        echo "Local image env file targets minikube profile ${expected_context}, but current kubectl context is ${current_context}." >&2
        exit 1
      fi
      ;;
    kind)
      local expected_name="${FLASHRL_LOCAL_CLUSTER_NAME:-kind}"
      local expected_context="kind-${expected_name}"
      if [[ "$current_context" != "$expected_context" ]]; then
        echo "Local image env file targets kind cluster ${expected_name}, but current kubectl context is ${current_context}." >&2
        exit 1
      fi
      ;;
    docker-desktop)
      if [[ "$current_context" != "docker-desktop" ]]; then
        echo "Local image env file targets docker-desktop, but current kubectl context is ${current_context}." >&2
        exit 1
      fi
      ;;
    *)
      echo "Unsupported or missing FLASHRL_LOCAL_CLUSTER_TYPE for local image mode: ${cluster_type:-<empty>}." >&2
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
  echo "run_platform_job.sh failed at line $line_no with exit code $exit_code" >&2
  if [[ -n "$ARTIFACT_DIR" ]]; then
    echo "Writing diagnostics to $ARTIFACT_DIR" >&2
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

  echo "Timed out waiting for ${kind}/${name} in namespace=${namespace}" >&2
  return 1
}

wait_for_job_completion() {
  local deadline=$((SECONDS + TIMEOUT_SECONDS))
  local status_json=""
  local job_phase=""
  local finished_at=""
  local last_completed_step=""
  local last_error=""

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
}
for key, value in fields.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
    )"

    job_phase="$JOB_PHASE"
    finished_at="$JOB_FINISHED_AT"
    last_completed_step="$JOB_LAST_COMPLETED_STEP"
    last_error="$JOB_LAST_ERROR"

    echo "phase=${job_phase:-<none>} lastCompletedStep=${last_completed_step:-0} finishedAt=${finished_at:-<none>}"
    if [[ -n "$last_error" ]]; then
      echo "lastError=$last_error"
    fi

    if [[ "$job_phase" == "Failed" ]]; then
      echo "FlashRLJob $JOB_NAME failed." >&2
      return 1
    fi
    if [[ -n "$finished_at" ]]; then
      return 0
    fi

    sleep "$POLL_SECONDS"
  done

  echo "Timed out waiting for FlashRLJob $JOB_NAME to finish." >&2
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
  usage >&2
  exit 1
fi

if [[ -z "$ARTIFACT_DIR" ]]; then
  ARTIFACT_DIR="$REPO_ROOT/logs/platform-runs/$DEFAULT_ARTIFACT_TAG"
fi

require_command kubectl
require_command python3

if [[ ! -f "$IMAGE_ENV_FILE" ]]; then
  echo "Image env file not found: $IMAGE_ENV_FILE" >&2
  exit 1
fi

trap 'on_error $? $LINENO' ERR
trap cleanup_tmp EXIT

log_section "Validate Cluster Connectivity"
run_cmd kubectl cluster-info >/dev/null
CURRENT_CONTEXT="$(capture_cmd kubectl config current-context)"
log_info "kubectl context: $CURRENT_CONTEXT"

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
fi

log_section "Delete Previous FlashRLJob"
run_cmd kubectl delete "$FLASHRL_JOB_RESOURCE" "$JOB_NAME" -n "$JOB_NAMESPACE" --ignore-not-found=true --wait=true

log_section "Apply FlashRLJob"
run_cmd kubectl apply -f "$PATCHED_JOB_YAML"

log_section "Wait for FlashRL Workloads"
wait_for_workload_rollout deployment "${JOB_NAME}-controller" "$JOB_NAMESPACE"
wait_for_workload_rollout statefulset "${JOB_NAME}-learner" "$JOB_NAMESPACE"
wait_for_workload_rollout deployment "${JOB_NAME}-serving" "$JOB_NAMESPACE"
wait_for_workload_rollout deployment "${JOB_NAME}-rollout" "$JOB_NAMESPACE"
wait_for_workload_rollout deployment "${JOB_NAME}-reward" "$JOB_NAMESPACE"

log_section "Wait for FlashRLJob Completion"
wait_for_job_completion

log_section "Verify Artifacts"
CONTROLLER_POD="$(capture_cmd kubectl get pods -n "$JOB_NAMESPACE" -l "flashrl.dev/job=$JOB_NAME,app.kubernetes.io/component=controller" -o jsonpath='{.items[0].metadata.name}')"
if [[ -z "$CONTROLLER_POD" ]]; then
  echo "Controller pod not found for $JOB_NAME" >&2
  exit 1
fi
log_info "Controller pod: $CONTROLLER_POD"

if [[ "$VERIFY_ARTIFACTS" == "1" ]]; then
  run_cmd_stdout_to_file "$ARTIFACT_DIR/checkpoints.txt" kubectl exec -n "$JOB_NAMESPACE" "$CONTROLLER_POD" -- sh -c 'find "$1" -type f | sort' sh "$CHECKPOINT_INSPECT_PATH"
  run_cmd_stdout_to_file "$ARTIFACT_DIR/weights.txt" kubectl exec -n "$JOB_NAMESPACE" "$CONTROLLER_POD" -- sh -c 'find "$1" -type f | sort' sh "$WEIGHT_INSPECT_PATH"

  if [[ ! -s "$ARTIFACT_DIR/checkpoints.txt" ]]; then
    echo "No checkpoint artifacts found under $CHECKPOINT_INSPECT_PATH" >&2
    exit 1
  fi
  if [[ ! -s "$ARTIFACT_DIR/weights.txt" ]]; then
    echo "No weight artifacts found under $WEIGHT_INSPECT_PATH" >&2
    exit 1
  fi
else
  echo "Skipping artifact file verification because storage paths are not inspectable filesystem paths."
fi

trap - ERR

log_section "FlashRLJob Finished"
log_info "Job: $JOB_NAME"
log_info "Namespace: $JOB_NAMESPACE"
log_info "Artifacts: $ARTIFACT_DIR"
echo
echo "Inspect:"
echo "  kubectl get $FLASHRL_JOB_RESOURCE $JOB_NAME -n $JOB_NAMESPACE -o yaml"
echo "  kubectl get pods -n $JOB_NAMESPACE -l flashrl.dev/job=$JOB_NAME"
echo "  kubectl logs -n $JOB_NAMESPACE -l flashrl.dev/job=$JOB_NAME --prefix=true"
echo
echo "Cleanup:"
echo "  bash scripts/cleanup_platform.sh --operator-namespace $OPERATOR_NAMESPACE"
