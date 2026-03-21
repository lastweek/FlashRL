#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE_MODE="${IMAGE_MODE:-registry}"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-}"
IMAGE_TAG="${IMAGE_TAG:-platform-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ENV_FILE="${OUTPUT_ENV_FILE:-$REPO_ROOT/logs/platform-images/$IMAGE_TAG/flashrl-images.env}"
LOCAL_CLUSTER_TYPE="${LOCAL_CLUSTER_TYPE:-}"
LOCAL_CLUSTER_PROFILE="${LOCAL_CLUSTER_PROFILE:-}"
LOCAL_CLUSTER_NAME="${LOCAL_CLUSTER_NAME:-}"

usage() {
  cat <<'EOF'
Usage:
  IMAGE_REGISTRY=<registry-prefix> bash scripts/build_platform_images.sh

Local mode examples:
  IMAGE_MODE=local LOCAL_CLUSTER_TYPE=minikube LOCAL_CLUSTER_PROFILE=minikube \
    bash scripts/build_platform_images.sh

  IMAGE_MODE=local LOCAL_CLUSTER_TYPE=kind LOCAL_CLUSTER_NAME=kind \
    bash scripts/build_platform_images.sh

  IMAGE_MODE=local LOCAL_CLUSTER_TYPE=docker-desktop \
    bash scripts/build_platform_images.sh
EOF
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

log_section() {
  printf '\n== %s ==\n' "$1"
}

require_command docker

case "$IMAGE_MODE" in
  registry)
    if [[ -z "$IMAGE_REGISTRY" ]]; then
      echo "IMAGE_REGISTRY is required when IMAGE_MODE=registry." >&2
      usage >&2
      exit 1
    fi
    ;;
  local)
    if [[ -z "$LOCAL_CLUSTER_TYPE" ]]; then
      echo "LOCAL_CLUSTER_TYPE is required when IMAGE_MODE=local." >&2
      usage >&2
      exit 1
    fi
    ;;
  *)
    echo "Unsupported IMAGE_MODE: $IMAGE_MODE" >&2
    usage >&2
    exit 1
    ;;
esac

mkdir -p "$(dirname "$OUTPUT_ENV_FILE")"

build_and_push_image() {
  local image_ref=$1
  local dockerfile=$2
  local label=$3
  log_section "Build and Push ${label}"
  docker build -f "$dockerfile" -t "$image_ref" .
  docker push "$image_ref"
}

build_image() {
  local image_ref=$1
  local dockerfile=$2
  local label=$3
  log_section "Build ${label}"
  docker build -f "$dockerfile" -t "$image_ref" .
}

if [[ "$IMAGE_MODE" == "registry" ]]; then
  FLASHRL_OPERATOR_IMAGE="${IMAGE_REGISTRY}/flashrl-operator:${IMAGE_TAG}"
  FLASHRL_RUNTIME_IMAGE="${IMAGE_REGISTRY}/flashrl-runtime:${IMAGE_TAG}"
  FLASHRL_SERVING_IMAGE="${IMAGE_REGISTRY}/flashrl-serving-vllm:${IMAGE_TAG}"
  FLASHRL_TRAINING_IMAGE="${IMAGE_REGISTRY}/flashrl-training-fsdp:${IMAGE_TAG}"

  build_and_push_image "$FLASHRL_OPERATOR_IMAGE" "docker/operator.Dockerfile" "flashrl-operator"
  build_and_push_image "$FLASHRL_RUNTIME_IMAGE" "docker/runtime.Dockerfile" "flashrl-runtime"
  build_and_push_image "$FLASHRL_SERVING_IMAGE" "docker/serving-vllm.Dockerfile" "flashrl-serving-vllm"
  build_and_push_image "$FLASHRL_TRAINING_IMAGE" "docker/training-fsdp.Dockerfile" "flashrl-training-fsdp"
else
  FLASHRL_OPERATOR_IMAGE="flashrl-operator:${IMAGE_TAG}"
  FLASHRL_RUNTIME_IMAGE="flashrl-runtime:${IMAGE_TAG}"
  FLASHRL_SERVING_IMAGE="flashrl-serving-vllm:${IMAGE_TAG}"
  FLASHRL_TRAINING_IMAGE="flashrl-training-fsdp:${IMAGE_TAG}"
  IMAGE_REGISTRY=""

  case "$LOCAL_CLUSTER_TYPE" in
    minikube)
      require_command kubectl
      require_command minikube
      if [[ -z "$LOCAL_CLUSTER_PROFILE" ]]; then
        LOCAL_CLUSTER_PROFILE="$(kubectl config current-context)"
      fi

      log_section "Build Images Into minikube Profile ${LOCAL_CLUSTER_PROFILE}"
      minikube -p "$LOCAL_CLUSTER_PROFILE" image build -t "$FLASHRL_OPERATOR_IMAGE" -f docker/operator.Dockerfile .
      minikube -p "$LOCAL_CLUSTER_PROFILE" image build -t "$FLASHRL_RUNTIME_IMAGE" -f docker/runtime.Dockerfile .
      minikube -p "$LOCAL_CLUSTER_PROFILE" image build -t "$FLASHRL_SERVING_IMAGE" -f docker/serving-vllm.Dockerfile .
      minikube -p "$LOCAL_CLUSTER_PROFILE" image build -t "$FLASHRL_TRAINING_IMAGE" -f docker/training-fsdp.Dockerfile .
      ;;
    kind)
      require_command kubectl
      require_command kind
      if [[ -z "$LOCAL_CLUSTER_NAME" ]]; then
        current_context="$(kubectl config current-context)"
        if [[ "$current_context" == kind-* ]]; then
          LOCAL_CLUSTER_NAME="${current_context#kind-}"
        else
          LOCAL_CLUSTER_NAME="kind"
        fi
      fi

      build_image "$FLASHRL_OPERATOR_IMAGE" "docker/operator.Dockerfile" "flashrl-operator"
      build_image "$FLASHRL_RUNTIME_IMAGE" "docker/runtime.Dockerfile" "flashrl-runtime"
      build_image "$FLASHRL_SERVING_IMAGE" "docker/serving-vllm.Dockerfile" "flashrl-serving-vllm"
      build_image "$FLASHRL_TRAINING_IMAGE" "docker/training-fsdp.Dockerfile" "flashrl-training-fsdp"

      log_section "Load Images Into kind Cluster ${LOCAL_CLUSTER_NAME}"
      kind load docker-image --name "$LOCAL_CLUSTER_NAME" "$FLASHRL_OPERATOR_IMAGE"
      kind load docker-image --name "$LOCAL_CLUSTER_NAME" "$FLASHRL_RUNTIME_IMAGE"
      kind load docker-image --name "$LOCAL_CLUSTER_NAME" "$FLASHRL_SERVING_IMAGE"
      kind load docker-image --name "$LOCAL_CLUSTER_NAME" "$FLASHRL_TRAINING_IMAGE"
      ;;
    docker-desktop)
      require_command kubectl
      build_image "$FLASHRL_OPERATOR_IMAGE" "docker/operator.Dockerfile" "flashrl-operator"
      build_image "$FLASHRL_RUNTIME_IMAGE" "docker/runtime.Dockerfile" "flashrl-runtime"
      build_image "$FLASHRL_SERVING_IMAGE" "docker/serving-vllm.Dockerfile" "flashrl-serving-vllm"
      build_image "$FLASHRL_TRAINING_IMAGE" "docker/training-fsdp.Dockerfile" "flashrl-training-fsdp"
      ;;
    *)
      echo "Unsupported LOCAL_CLUSTER_TYPE: $LOCAL_CLUSTER_TYPE" >&2
      usage >&2
      exit 1
      ;;
  esac
fi

log_section "Write Image Environment File"
{
  printf 'FLASHRL_OPERATOR_IMAGE=%q\n' "$FLASHRL_OPERATOR_IMAGE"
  printf 'FLASHRL_RUNTIME_IMAGE=%q\n' "$FLASHRL_RUNTIME_IMAGE"
  printf 'FLASHRL_SERVING_IMAGE=%q\n' "$FLASHRL_SERVING_IMAGE"
  printf 'FLASHRL_TRAINING_IMAGE=%q\n' "$FLASHRL_TRAINING_IMAGE"
  printf 'FLASHRL_IMAGE_MODE=%q\n' "$IMAGE_MODE"
  printf 'FLASHRL_IMAGE_REGISTRY=%q\n' "$IMAGE_REGISTRY"
  printf 'FLASHRL_IMAGE_TAG=%q\n' "$IMAGE_TAG"
  printf 'FLASHRL_LOCAL_CLUSTER_TYPE=%q\n' "$LOCAL_CLUSTER_TYPE"
  printf 'FLASHRL_LOCAL_CLUSTER_PROFILE=%q\n' "$LOCAL_CLUSTER_PROFILE"
  printf 'FLASHRL_LOCAL_CLUSTER_NAME=%q\n' "$LOCAL_CLUSTER_NAME"
} > "$OUTPUT_ENV_FILE"

echo "Wrote image refs to $OUTPUT_ENV_FILE"
echo "Next step:"
echo "  bash scripts/run_platform_job.sh --config flashrl/examples/math/config.yaml --image-env-file $OUTPUT_ENV_FILE"
