#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_ROOT="${1:-outputs/1}"
FORCE="${2:-}"

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
  echo "Usage: commands/denoise.sh [outputs/1|runs/1|artifact_dir] [--force]" >&2
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

for artifact_dir in "${ARTIFACT_DIRS[@]}"; do
  output_path="${artifact_dir}/denoised.mp4"
  if [[ -f "${output_path}" && "${FORCE}" != "--force" ]]; then
    echo "Skipping ${artifact_dir} (denoised.mp4 already exists)"
    continue
  fi

  echo "Denoising ${artifact_dir}"
  python3 scripts/denoise_debug_forward.py --input "${artifact_dir}" --output "${output_path}"
done
