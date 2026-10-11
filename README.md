# AI Images — CPU-only image, GIF, and video generation

This repository runs open-weight generation models on GitHub Actions using **CPU only**. All three workflows accept prompts in **Hebrew or English**. If Hebrew characters are detected, the prompt is translated locally to English before generation.

## Important limits

GitHub-hosted runners have limited CPU, RAM, and SSD. Video diffusion on CPU is extremely slow and may exceed RAM, disk, or job-time limits; the video workflow is experimental and is not guaranteed to finish.

## Workflows

- **Generate image (CPU)** — Stable Diffusion 1.5, one 512×512 PNG.
- **Generate animated GIF (CPU)** — generates one AI image and creates a looping GIF with a gentle zoom animation. The GIF does not generate new scene content in every frame.
- **Generate video (CPU, experimental)** — ModelScope text-to-video 1.7B, configured for a very short 256×256 clip and few steps. Expect very slow generation and possible out-of-memory, disk, or time-limit failures.

## Run it

1. Open the **Actions** tab.
2. Select **Generate image (CPU)**, **Generate animated GIF (CPU)**, or **Generate video (CPU, experimental)**.
3. Click **Run workflow** and enter a prompt in Hebrew or English.
4. When it finishes, download the corresponding artifact from the run summary: `generated-image`, `generated-gif`, or `generated-video`.

## Hebrew prompt handling

Hebrew is detected automatically. The workflow uses the local Hugging Face translation model `Helsinki-NLP/opus-mt-tc-big-he-en` to translate Hebrew prompts to English, since the image and video generation models work best with English prompts. English prompts are passed through unchanged. The translation model is downloaded on the first run and adds extra setup time and disk use; translation quality can vary, especially for complex or ambiguous descriptions.

No API key is required. Model weights are downloaded during the run; generated media and model weights are not committed into Git.

## Model notes

- Image and GIF artwork: `stable-diffusion-v1-5/stable-diffusion-v1-5` through Hugging Face Diffusers.
- Hebrew-to-English translation: `Helsinki-NLP/opus-mt-tc-big-he-en`, a local model under CC-BY-4.0.
- Video: `damo-vilab/text-to-video-ms-1.7b` through Diffusers. The upstream model card states English-only input and a CC-BY-NC-ND license, which restricts use and redistribution; check the model card before using or sharing outputs.

## Troubleshooting

- **Out of memory:** the hosted runner has a fixed RAM limit; fewer steps do not necessarily reduce model-loading memory enough.
- **Disk space:** translation and generation model weights together use additional SSD space.
- **Timeout:** video diffusion may be too slow on CPU to finish within a hosted job.
- **Model download failure:** check Hugging Face availability and the workflow logs.

This is an experimental project, not a promise of fast or reliable generation on a free runner.
