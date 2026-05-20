#!/usr/bin/env python3
"""Prepull mimic-video deployment assets before starting the gRPC server."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_SRC = REPO_ROOT / "lerobot_policy_mimic_video/src"
DEFAULT_DEPLOY_REPO = "rl26-world-models/mimic-finetune"
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))


def _load_training_config_module():
    module_path = PLUGIN_SRC / "lerobot_policy_mimic_video/training_config.py"
    spec = importlib.util.spec_from_file_location("mimic_video_training_config", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_training_config = _load_training_config_module()
build_deployment_config = _training_config.build_deployment_config
maybe_download_file = _training_config.maybe_download_file


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value


def _path_env(name: str) -> Path | None:
    value = _env(name)
    return Path(value).expanduser() if value else None


def _model_repo_id(label: str) -> str | None:
    return _env(f"{label.upper()}_MODEL_HF_REPO_ID") or _env("MIMIC_VIDEO_HF_REPO_ID") or DEFAULT_DEPLOY_REPO


def _stats_repo_id() -> str | None:
    return _env("HF_STATS_REPO_ID") or _env("MIMIC_VIDEO_HF_REPO_ID") or DEFAULT_DEPLOY_REPO


def _prepull_text_encoder(*, cache_dir: Path) -> Path | None:
    repo_id = _env("TEXT_ENCODER_HF_REPO_ID") or _env("MIMIC_VIDEO_HF_REPO_ID") or DEFAULT_DEPLOY_REPO
    allow_pattern = _env("TEXT_ENCODER_HF_ALLOW_PATTERN") or _env("TEXT_ENCODER_HF_FILENAME") or "text_encoder/**"
    revision = _env("TEXT_ENCODER_HF_REVISION")
    from huggingface_hub import snapshot_download

    target_dir = _path_env("TEXT_ENCODER_CACHE_DIR") or (cache_dir / "text_encoder")
    kwargs = {
        "repo_id": repo_id,
        "repo_type": _env("TEXT_ENCODER_HF_REPO_TYPE") or "model",
        "revision": revision or "main",
        "local_dir": str(target_dir),
    }
    if allow_pattern:
        kwargs["allow_patterns"] = [allow_pattern]
    snapshot_download(**kwargs)
    return target_dir.resolve()


def _resolve_or_download(
    *,
    label: str,
    explicit_path: Path | None,
    config_path: Path | None,
    cache_dir: Path,
    repo_id: str | None,
    filename: str | None,
    revision: str | None,
    repo_type: str,
) -> Path:
    path = explicit_path or config_path
    if path is not None and path.exists():
        return path.resolve()
    if filename:
        cache_path = path if path is not None and not path.is_absolute() else cache_dir / Path(filename).name
        return maybe_download_file(
            path=cache_path,
            hf_repo_id=repo_id,
            hf_filename=filename,
            hf_revision=revision,
            repo_type=repo_type,
        )
    if path is not None:
        raise FileNotFoundError(
            f"{label} checkpoint path from config/env does not exist: {path}\n"
            f"Export this checkpoint to the container/mount, or set {label.upper()}_MODEL_HF_REPO_ID "
            f"and {label.upper()}_MODEL_HF_FILENAME so it can be prepulled."
        )
    raise FileNotFoundError(
        f"{label} checkpoint is not specified. Put it in TRAINING_CONFIG or set "
        f"{label.upper()}_MODEL_PATH / {label.upper()}_MODEL_HF_REPO_ID + {label.upper()}_MODEL_HF_FILENAME."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, default=_path_env("TRAINING_CONFIG"))
    parser.add_argument("--checkpoint-cache-dir", type=Path, default=_path_env("CHECKPOINT_CACHE_DIR") or Path("/checkpoints"))
    parser.add_argument("--write-env", type=Path, default=_path_env("PREPULL_ENV_FILE"))
    parser.add_argument("--repo-id", default=_env("MIMIC_VIDEO_HF_REPO_ID") or DEFAULT_DEPLOY_REPO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.training_config is None:
        raise ValueError("TRAINING_CONFIG or --training-config is required.")
    os.environ.setdefault("MIMIC_VIDEO_HF_REPO_ID", args.repo_id)

    stats_path = _path_env("STATS_PATH")
    hf_stats_filename = _env("HF_STATS_FILENAME")
    if stats_path is None and hf_stats_filename:
        stats_path = args.checkpoint_cache_dir.expanduser() / "dataset_statistics" / Path(hf_stats_filename).name

    deployment = build_deployment_config(
        training_config=args.training_config,
        stats_path=stats_path,
        action_data_dir=_path_env("ACTION_DATA_DIR"),
        hf_stats_repo_id=_stats_repo_id(),
        hf_stats_filename=hf_stats_filename,
        hf_stats_revision=_env("HF_STATS_REVISION"),
    )
    cache_dir = args.checkpoint_cache_dir.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    video_path = _resolve_or_download(
        label="video",
        explicit_path=_path_env("VIDEO_MODEL_PATH"),
        config_path=deployment.video_model_path,
        cache_dir=cache_dir,
        repo_id=_model_repo_id("video"),
        filename=_env("VIDEO_MODEL_HF_FILENAME"),
        revision=_env("VIDEO_MODEL_HF_REVISION"),
        repo_type=_env("VIDEO_MODEL_HF_REPO_TYPE") or "model",
    )
    action_path = _resolve_or_download(
        label="action",
        explicit_path=_path_env("ACTION_MODEL_PATH"),
        config_path=deployment.action_model_path,
        cache_dir=cache_dir,
        repo_id=_model_repo_id("action"),
        filename=_env("ACTION_MODEL_HF_FILENAME"),
        revision=_env("ACTION_MODEL_HF_REVISION"),
        repo_type=_env("ACTION_MODEL_HF_REPO_TYPE") or "model",
    )
    text_encoder_path = None
    if _env("PREPULL_TEXT_ENCODER") == "1":
        text_encoder_path = _prepull_text_encoder(cache_dir=cache_dir)

    lines = [
        f"VIDEO_MODEL_PATH={video_path}",
        f"ACTION_MODEL_PATH={action_path}",
        f"STATS_PATH={deployment.stats_path}",
    ]
    if text_encoder_path is not None:
        lines.append(f"TEXT_ENCODER_CACHE_DIR={text_encoder_path}")
    print("\n".join(lines))
    if args.write_env is not None:
        args.write_env.parent.mkdir(parents=True, exist_ok=True)
        args.write_env.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
