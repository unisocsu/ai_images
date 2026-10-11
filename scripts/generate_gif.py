import math
import os
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image

from prompt_translation import translate_prompt_if_hebrew

PROMPT = os.environ.get(
    "PROMPT",
    "A tiny wooden cabin in a magical forest, golden sunbeams, fireflies, cinematic, detailed",
)
STEPS = int(os.environ.get("STEPS", "5"))
FRAMES = int(os.environ.get("FRAMES", "16"))
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SIZE = 384


def main() -> None:
    output = Path("output")
    output.mkdir(parents=True, exist_ok=True)

    prompt = translate_prompt_if_hebrew(PROMPT)

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    print(f"Loading {MODEL_ID} on CPU. The first run downloads model weights.")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()

    generator = torch.Generator(device="cpu").manual_seed(42)
    with torch.inference_mode():
        base = pipe(
            prompt=prompt,
            width=SIZE,
            height=SIZE,
            num_inference_steps=STEPS,
            guidance_scale=7.0,
            generator=generator,
        ).images[0].convert("RGB")

    # Animate a gentle zoom-in/zoom-out on the generated artwork.
    # This creates a lightweight animated GIF without running a separate video model.
    frames = []
    for i in range(FRAMES):
        phase = i / max(1, FRAMES - 1)
        zoom = 1.0 + 0.10 * (0.5 - 0.5 * math.cos(2 * math.pi * phase))
        enlarged = base.resize(
            (round(SIZE * zoom), round(SIZE * zoom)),
            Image.Resampling.LANCZOS,
        )
        left = (enlarged.width - SIZE) // 2
        top = (enlarged.height - SIZE) // 2
        frame = enlarged.crop((left, top, left + SIZE, top + SIZE))
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))

    destination = output / "animation.gif"
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"Saved {destination.resolve()} ({FRAMES} frames)")


if __name__ == "__main__":
    main()
