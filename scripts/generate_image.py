import os
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

from prompt_translation import translate_prompt_if_hebrew

PROMPT = os.environ.get("PROMPT", "A tiny wooden cabin in a snowy forest at sunrise, cinematic photography")
STEPS = int(os.environ.get("STEPS", "10"))
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def main() -> None:
    output = Path("output")
    output.mkdir(parents=True, exist_ok=True)

    # Translate Hebrew locally before loading the larger image model.
    prompt = translate_prompt_if_hebrew(PROMPT)

    # CPU-only inference: do not attempt to use CUDA even if a package detects it.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    print(f"Loading {MODEL_ID} on CPU; first run downloads model weights.")
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
        image = pipe(
            prompt=prompt,
            width=512,
            height=512,
            num_inference_steps=STEPS,
            guidance_scale=7.0,
            generator=generator,
        ).images[0]

    destination = output / "image.png"
    image.save(destination)
    print(f"Saved {destination.resolve()}")


if __name__ == "__main__":
    main()
