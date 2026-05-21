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
    parser.add_argument("--input", type=Path, required=True, help="Debug output directory containing video_denoiser_output.pt.")
    parser.add_argument("--output", type=Path, default=None, help="Path for the decoded video.")
    parser.add_argument("--dit_path", type=str, default=None, help="Override video DiT checkpoint path.")
    parser.add_argument("--guidance", type=float, default=0.0, help="Classifier-free guidance for resumed denoising.")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS.")
    parser.add_argument("--use_cuda_graphs", action="store_true")
    return parser.parse_args()


def load_seed_frames(seed_frame_dir: Path) -> torch.Tensor:
    frame_paths = sorted(seed_frame_dir.glob("*.png"), key=lambda path: int(path.stem))
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


def main() -> None:
    args = parse_args()
    input_dir = args.input
    artifact_path = input_dir / "video_denoiser_output.pt"
    output_path = args.output or (input_dir / "denoised.mp4")

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    vid_input = load_seed_frames(input_dir / "seed_frames")
    pipe = load_video_pipeline(artifact, args.dit_path).cuda().eval()

    with torch.no_grad():
        video = pipe.continue_denoising_from_debug_artifact(
            artifact=artifact,
            vid_input=vid_input,
            guidance=args.guidance,
            use_cuda_graphs=args.use_cuda_graphs,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_or_video(video, str(output_path), fps=args.fps)
    print(f"Saved denoised video to {output_path}")


if __name__ == "__main__":
    main()
