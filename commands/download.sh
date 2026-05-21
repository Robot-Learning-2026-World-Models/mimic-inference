if [[ -f .env ]]; then
  source .env
fi

# Keep downloads in the repo-local cache, but do not strand auth in the default
# Hugging Face cache. `hf auth login` normally writes ~/.cache/huggingface/token.
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  default_hf_token_path="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
  if [[ -r "$default_hf_token_path" ]]; then
    export HF_TOKEN="$(<"$default_hf_token_path")"
  fi
fi

export HF_HOME="${HF_HOME:-$PWD/hf-cache}"
if ! mkdir -p "$HF_HOME" 2>/dev/null || [[ ! -w "$HF_HOME" ]]; then
  export HF_HOME="$PWD/.hf-cache"
  mkdir -p "$HF_HOME"
fi
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=300

export TRAINING_CONFIG="$PWD/mimic-video/configs/so101/idm_train_dual_task_low_rank.yaml"
export CHECKPOINT_CACHE_DIR="$PWD/checkpoints"
export MIMIC_VIDEO_HF_REPO_ID="mimicinferencegobrrrr/mimic-finetune"
export VIDEO_MODEL_HF_FILENAME="video_backbone/backbone4500.pt"
export ACTION_MODEL_HF_FILENAME="action_decoder/idm_110.pt"
export HF_STATS_FILENAME="dataset_statistics/idm_110m_dataset_stats.json"
export HF_STATS_REPO_TYPE="model"
export PREPULL_TEXT_ENCODER=0
export PYTHONPATH="$PWD/lerobot_policy_mimic_video/src:$PWD/mimic-video/model"

source mimic-video/model/.venv/bin/activate

mimic-video/model/.venv/bin/python scripts/prepull_mimic_video_assets.py \
  --write-env="$PWD/checkpoints/mimic_video_assets.env"

mimic-video/model/.venv/bin/hf download "$MIMIC_VIDEO_HF_REPO_ID" \
  video_backbone/tokenizer/tokenizer.pth \
  --repo-type model \
  --local-dir "$PWD/checkpoints"
