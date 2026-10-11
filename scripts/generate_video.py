import os
from pathlib import Path

import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

PROMPT = os.environ.get("PROMPT", "A small red train moving slowly through a green countryside, cinematic wide shot")
STEPS = int(os.environ.get("STEPS", "4"))
# Pin a smaller legacy LTX checkpoint rather than the newer 13B family.
MODEL_ID = "a-r-r-o-w/LTX-Video-0.9.1-diffusers"

def main() -> None:
    output = Path("output")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    print(f"Loading {MODEL_ID} on CPU. This model is large and may exceed the runner's RAM or disk.")
    pipe = LTXPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    pipe = pipe.to("cpu")
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    # Very short, low-resolution test settings intended to minimize compute.
    with torch.inference_mode():
        result = pipe(
            prompt=PROMPT,
            negative_prompt="blurry, jittery motion, distorted frames, low quality",
            width=256,
            height=256,
            num_frames=9,
            frame_rate=8,
            num_inference_steps=STEPS,
            guidance_scale=3.0,
            max_sequence_length=64,
        )

    frames = result.frames[0]
    destination = output / "video.mp4"
    export_to_video(frames, str(destination), fps=8)
    print(f"Saved {destination.resolve()}")

if __name__ == "__main__":
    main()