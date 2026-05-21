from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "mimic-video" / "model"
sys.path.insert(0, str(MODEL_ROOT))

from cosmos_predict2.configs.config import make_config  # noqa: E402
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline  # noqa: E402
from imaginaire.utils.config_helper import override  # noqa: E402
from imaginaire.utils.io import save_image_or_video  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue a first-step video denoiser debug artifact.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        nargs="+",
        help="One or more debug output directories containing video_denoiser_output.pt.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Path for the decoded video.")
    parser.add_argument("--dit_path", type=str, default=None, help="Override video DiT checkpoint path.")
    parser.add_argument("--guidance", type=float, default=0.0, help="Classifier-free guidance for resumed denoising.")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS.")
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=1,
        help="Number of artifacts to denoise per GPU batch.",
    )
    parser.add_argument("--use_cuda_graphs", action="store_true")
    return parser.parse_args()


def load_seed_frames(seed_frame_dir: Path) -> torch.Tensor:
    frame_paths = sorted(seed_frame_dir.glob("*.png"), key=lambda path: int(path.stem.split("_", 1)[0]))
    if not frame_paths:
        raise FileNotFoundError(f"No seed frames found in {seed_frame_dir}")

    frames = []
    for path in frame_paths:
        frame = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        frames.append(frame)

    video = np.stack(frames, axis=0)
    video = np.moveaxis(video, -1, 1)
    video = 2.0 * (video / 255.0 - 0.5)
    return torch.from_numpy(video).permute(1, 0, 2, 3).unsqueeze(0).cuda().to(dtype=torch.bfloat16)


def load_video_pipeline(artifact: dict, dit_path: str | None) -> Video2WorldPipeline:
    experiment_name = artifact.get("experiment_name")
    if not experiment_name:
        raise KeyError("Artifact is missing experiment_name; pass an artifact produced by debug inference.")

    checkpoint_path = dit_path or artifact.get("video_model_path")
    if not checkpoint_path:
        raise KeyError("Artifact is missing video_model_path; pass --dit_path.")

    cfg = make_config()
    cfg = override(cfg, ["--", f"experiment={experiment_name}"])
    cfg.model.config.video_pipe_config.guardrail_config.enabled = False
    if "video_pipe_state_t" in artifact:
        cfg.model.config.video_pipe_config.state_t = int(artifact["video_pipe_state_t"])

    use_text_encoder = artifact.get("prompt_embedding") is None
    return Video2WorldPipeline.from_config(
        config=cfg.model.config.video_pipe_config,
        dit_path=str(checkpoint_path),
        use_text_encoder=use_text_encoder,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=False,
    )


def batched(items: list[Path], batch_size: int) -> list[list[Path]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def load_artifact(input_dir: Path) -> dict:
    artifact_path = input_dir / "video_denoiser_output.pt"
    return torch.load(artifact_path, map_location="cpu", weights_only=False)


def assert_batch_compatible(reference: dict, artifact: dict, input_dir: Path) -> None:
    keys = ["num_sampling_steps", "scheduler_step", "experiment_name", "video_model_path", "video_pipe_state_t"]
    for key in keys:
        if reference.get(key) != artifact.get(key):
            raise ValueError(f"{input_dir} has incompatible {key}: {artifact.get(key)!r} != {reference.get(key)!r}")

    if isinstance(reference.get("sigmas"), torch.Tensor) and isinstance(artifact.get("sigmas"), torch.Tensor):
        if not torch.equal(reference["sigmas"], artifact["sigmas"]):
            raise ValueError(f"{input_dir} has incompatible scheduler sigmas.")


def merge_artifacts(artifacts: list[dict]) -> dict:
    merged = dict(artifacts[0])
    merged["sample_before_step"] = torch.cat([artifact["sample_before_step"] for artifact in artifacts], dim=0)
    merged["x0_pred"] = torch.cat([artifact["x0_pred"] for artifact in artifacts], dim=0)

    x0_prevs = [artifact.get("x0_prev") for artifact in artifacts]
    merged["x0_prev"] = torch.cat(x0_prevs, dim=0) if all(isinstance(x, torch.Tensor) for x in x0_prevs) else None

    prompt_embeddings = [artifact.get("prompt_embedding") for artifact in artifacts]
    if all(isinstance(embedding, torch.Tensor) for embedding in prompt_embeddings):
        merged["prompt_embedding"] = torch.cat(prompt_embeddings, dim=0)
    else:
        merged["prompt_embedding"] = None
        merged["prompt"] = [artifact.get("prompt", "") for artifact in artifacts]

    return merged


def denoise_batch(
    *,
    pipe: Video2WorldPipeline,
    input_dirs: list[Path],
    artifacts: list[dict],
    args: argparse.Namespace,
) -> None:
    reference = artifacts[0]
    for input_dir, artifact in zip(input_dirs[1:], artifacts[1:], strict=True):
        assert_batch_compatible(reference, artifact, input_dir)

    vid_input = torch.cat([load_seed_frames(input_dir / "seed_frames") for input_dir in input_dirs], dim=0)
    merged_artifact = merge_artifacts(artifacts)

    with torch.no_grad():
        videos = pipe.continue_denoising_from_debug_artifact(
            artifact=merged_artifact,
            vid_input=vid_input,
            prompt=merged_artifact.get("prompt"),
            guidance=args.guidance,
            use_cuda_graphs=args.use_cuda_graphs,
        )

    for idx, input_dir in enumerate(input_dirs):
        output_path = args.output if len(input_dirs) == 1 and args.output is not None else input_dir / "denoised.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image_or_video(videos[idx : idx + 1], str(output_path), fps=args.fps)
        print(f"Saved denoised video to {output_path}")


def main() -> None:
    args = parse_args()
    input_dirs = args.input
    if args.output is not None and len(input_dirs) > 1:
        raise ValueError("--output can only be used with one --input directory.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")

    first_artifact = load_artifact(input_dirs[0])
    pipe = load_video_pipeline(first_artifact, args.dit_path).cuda().eval()

    for input_batch in batched(input_dirs, args.batch_size):
        artifacts = [first_artifact if input_dir == input_dirs[0] else load_artifact(input_dir) for input_dir in input_batch]
        for input_dir, artifact in zip(input_batch, artifacts, strict=True):
            assert_batch_compatible(first_artifact, artifact, input_dir)
        print(f"Denoising GPU batch of {len(input_batch)}: {', '.join(str(path) for path in input_batch)}")
        denoise_batch(pipe=pipe, input_dirs=input_batch, artifacts=artifacts, args=args)


if __name__ == "__main__":
    main()
