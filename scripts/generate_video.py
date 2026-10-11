import os
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline

from prompt_translation import translate_prompt_if_hebrew
from diffusers.utils import export_to_video

PROMPT = os.environ.get("PROMPT", "A small red train moving slowly through a green countryside, cinematic wide shot")
STEPS = int(os.environ.get("STEPS", "4"))
MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"


def main() -> None:
    output = Path("output")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    # The video model expects English; translate Hebrew prompts locally first.
    prompt = translate_prompt_if_hebrew(PROMPT)

    print(f"Loading {MODEL_ID} on CPU. This is experimental and may be slow or exceed runner limits.")
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

    # Keep the first test intentionally small. CPU inference can still take a long time.
    generator = torch.Generator(device="cpu").manual_seed(42)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            height=256,
            width=256,
            num_frames=8,
            num_inference_steps=STEPS,
            guidance_scale=7.5,
            generator=generator,
        )

    frames = result.frames[0]
    destination = output / "video.mp4"
    export_to_video(frames, str(destination), fps=8)
    print(f"Saved {destination.resolve()}")


if __name__ == "__main__":
    main()
