#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_ROOT="outputs"
FORCE=0
BATCH_SIZE="${DENOISE_BATCH_SIZE:-2}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --batch-size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --batch-size)
      if [[ "$#" -lt 2 ]]; then
        echo "--batch-size requires a value, e.g. --batch-size 2" >&2
        exit 1
      fi
      BATCH_SIZE="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: commands/denoise.sh [outputs|outputs/1|artifact_dir] [--batch-size N] [--force]"
      exit 0
      ;;
    *)
      INPUT_ROOT="$1"
      shift
      ;;
  esac
done

if [[ -f "${REPO_ROOT}/.env" ]]; then
  source "${REPO_ROOT}/.env"
fi

export PATH="${REPO_ROOT}/mimic-video/model/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/lerobot_policy_mimic_video/src:${REPO_ROOT}/mimic-video/model"
export COSMOS_PREDICT2_ARGS="--checkpoints ${REPO_ROOT}/checkpoints"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/hf-cache}"
export TOKENIZERS_PARALLELISM=false

source "${REPO_ROOT}/mimic-video/model/.venv/bin/activate"

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "Input root does not exist: ${INPUT_ROOT}" >&2
  echo "Usage: commands/denoise.sh [outputs|outputs/1|artifact_dir] [--batch-size N] [--force]" >&2
  exit 1
fi

if ! [[ "${BATCH_SIZE}" =~ ^[0-9]+$ ]] || [[ "${BATCH_SIZE}" -lt 1 ]]; then
  echo "--batch-size must be a positive integer, got: ${BATCH_SIZE}" >&2
  exit 1
fi

mapfile -t ARTIFACT_DIRS < <(
  find "${INPUT_ROOT}" -type f -name video_denoiser_output.pt -printf '%h\n' | sort -V
)

if [[ "${#ARTIFACT_DIRS[@]}" -eq 0 ]]; then
  echo "No video_denoiser_output.pt files found under ${INPUT_ROOT}" >&2
  exit 1
fi

echo "Found ${#ARTIFACT_DIRS[@]} debug artifacts under ${INPUT_ROOT}"
echo "GPU batch size: ${BATCH_SIZE}"

completed=0
skipped=0
PENDING_DIRS=()
for artifact_dir in "${ARTIFACT_DIRS[@]}"; do
  output_path="${artifact_dir}/denoised.mp4"
  if [[ -s "${output_path}" && "${FORCE}" -eq 0 ]]; then
    echo "Skipping ${artifact_dir} (denoised.mp4 already exists)"
    skipped=$((skipped + 1))
    continue
  fi

  PENDING_DIRS+=("${artifact_dir}")
  completed=$((completed + 1))
done

if [[ "${#PENDING_DIRS[@]}" -gt 0 ]]; then
  python3 scripts/denoise_debug_forward.py --batch-size "${BATCH_SIZE}" --input "${PENDING_DIRS[@]}"
fi

echo "Done. Denoised ${completed}, skipped ${skipped}."
