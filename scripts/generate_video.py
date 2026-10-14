import os
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline
from diffusers.utils import export_to_video

from prompt_translation import translate_prompt_if_hebrew

PROMPT = os.environ.get(
    "PROMPT",
    "A small red train moving slowly through a green countryside, cinematic wide shot",
)
SECONDS = int(os.environ.get("SECONDS", "2"))
STEPS = int(os.environ.get("STEPS", "12"))
FPS = int(os.environ.get("FPS", "8"))
MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"


def main() -> None:
    if SECONDS not in (2, 3, 4):
        raise ValueError("SECONDS must be 2, 3, or 4.")
    if FPS not in (4, 8):
        raise ValueError("FPS must be 4 or 8.")
    if STEPS not in (8, 12, 16):
        raise ValueError("STEPS must be 8, 12, or 16.")

    num_frames = SECONDS * FPS
    output = Path("output")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    # The video model expects English; translate Hebrew prompts locally first.
    prompt = translate_prompt_if_hebrew(PROMPT)

    # Duration and FPS are user-configurable; more frames and inference steps
    # increase the amount of CPU work substantially.
    print(
        f"Loading {MODEL_ID} on CPU; generating {num_frames} frames "
        f"({SECONDS} seconds) at {FPS} fps with {STEPS} inference steps."
    )
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cpu")
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    generator = torch.Generator(device="cpu").manual_seed(42)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            height=256,
            width=256,
            num_frames=num_frames,
            num_inference_steps=STEPS,
            guidance_scale=7.5,
            generator=generator,
        )

    frames = result.frames[0]
    if len(frames) < num_frames:
        raise RuntimeError(
            f"Video model returned only {len(frames)} frames; expected {num_frames}. "
            "Refusing to publish an unexpectedly short clip."
        )

    destination = output / "video.mp4"
    export_to_video(frames, str(destination), fps=FPS)
    size_bytes = destination.stat().st_size
    if size_bytes < 10_000:
        raise RuntimeError(
            f"Generated MP4 is suspiciously small ({size_bytes} bytes); "
            "refusing to treat it as a valid video."
        )

    print(
        f"Saved {destination.resolve()} — {len(frames)} frames at {FPS} fps "
        f"({len(frames) / FPS:.1f} seconds), {size_bytes:,} bytes"
    )


if __name__ == "__main__":
    main()
