# AI Images — CPU-only image, GIF, and video generation

This repository runs open-weight generation models on GitHub Actions using **CPU only**. Start jobs from the **Actions** tab; generated files are uploaded as downloadable artifacts.

## Important limits

GitHub-hosted public-repository runners have limited CPU, RAM, and SSD. Video diffusion on CPU is extremely slow and may exceed RAM, disk, or job-time limits; the video workflow is experimental and is not guaranteed to finish.

## Workflows

- **Generate image (CPU)** — Stable Diffusion 1.5, one 512×512 PNG.
- **Generate animated GIF (CPU)** — generates one AI image and creates a looping GIF with a gentle zoom animation. The GIF workflow uses a low number of diffusion steps by default to limit CPU time; it does not generate new scene content in every frame.
- **Generate video (CPU, experimental)** — ModelScope text-to-video 1.7B, configured for a very short 256×256 clip and few steps. Expect very slow generation and possible out-of-memory, disk, or time-limit failures.

## Run it

1. Open the **Actions** tab.
2. Select **Generate image (CPU)**, **Generate animated GIF (CPU)**, or **Generate video (CPU, experimental)**.
3. Click **Run workflow**, enter a prompt, choose the available quality/frame options, and start the run.
4. When it finishes, download the corresponding artifact from the run summary: `generated-image`, `generated-gif`, or `generated-video`.

No API key is required for the configured public model repositories. Model weights are downloaded during the run, so the first run can take a while. Generated media and model weights are not committed into Git.

## Model notes

- Image and GIF artwork: `stable-diffusion-v1-5/stable-diffusion-v1-5` through Hugging Face Diffusers.
- Video: `damo-vilab/text-to-video-ms-1.7b` through Diffusers. The upstream model card states English-only input and a CC-BY-NC-ND license, which restricts use and redistribution; check the model card before using or sharing outputs. This workflow is a best-effort CPU experiment, not a promise of successful generation.

## Troubleshooting

- **Out of memory:** the hosted runner has a fixed RAM limit; fewer steps do not necessarily reduce model-loading memory enough.
- **Disk space:** model weights may use a substantial fraction of the runner's limited SSD.
- **Timeout:** video diffusion may be too slow on CPU to finish within a hosted job.
- **Model download failure:** check Hugging Face availability and the workflow logs.

This is an experimental project, not a promise of fast or reliable generation on a free runner.
