# AI Images — CPU-only image and video generation

This repository runs open-weight generation models on GitHub Actions using **CPU only**. Start jobs from the **Actions** tab; generated files are uploaded as downloadable artifacts.

## Important limits

GitHub-hosted public-repository runners currently provide a standard Linux runner with 4 CPUs and 16 GB RAM, plus limited local SSD. Hosted jobs have a maximum runtime. Video diffusion on CPU is extremely slow and may exceed RAM, disk, or job-time limits; the video workflow is experimental and is not guaranteed to finish.

## Workflows

- **Generate image (CPU)** — Stable Diffusion 1.5, one 512×512 PNG, 10 inference steps by default.
- **Generate video (CPU, experimental)** — an older LTX-Video Diffusers checkpoint, using a very short low-resolution clip and few steps. Expect very slow generation and possible out-of-memory, disk, or time-limit failures.

## Run it

1. Open the **Actions** tab.
2. Select **Generate image (CPU)** or **Generate video (CPU, experimental)**.
3. Click **Run workflow**, enter a prompt, and start the run.
4. When it finishes, download the generated-image or generated-video artifact from the run summary.

No API key is required for the configured public model repositories. Model weights are downloaded during the run, so the first run can take a while. The workflow does not commit generated media or model weights into Git.

## Model notes

- Image: `stable-diffusion-v1-5/stable-diffusion-v1-5` through Hugging Face Diffusers.
- Video: `a-r-r-o-w/LTX-Video-0.9.1-diffusers` through Diffusers. The video workflow is a best-effort CPU experiment; model/runtime compatibility can change as upstream packages evolve. Check the upstream model card and license before redistributing outputs or weights.

## Troubleshooting

- **Out of memory:** the hosted runner has a fixed RAM limit; fewer steps do not necessarily reduce model-loading memory enough.
- **Disk space:** runners have limited SSD; model weights may use a substantial fraction of it.
- **Timeout:** video diffusion may be too slow on CPU to finish within a hosted job.
- **Model download failure:** check Hugging Face availability and the workflow logs.

This is an experimental project, not a promise of fast or reliable video generation on a free runner.