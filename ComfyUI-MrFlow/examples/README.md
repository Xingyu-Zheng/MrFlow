# Example Workflows

This folder contains the Qwen MrFlow workflows shipped with the extension.

## Files

- `qwen_mrflow_workflow.json`
- `qwen_mrflow_api.json`

Both files follow the same four-stage path:

1. low-resolution generation
2. pixel-space super-resolution
3. latent re-encode
4. high-resolution direct-sigma refine

## Model files

The workflow expects these files to be present in the standard ComfyUI model folders:

- `qwen_image_bf16.safetensors` in `models/diffusion_models/`
- `qwen_2.5_vl_7b.safetensors` in `models/text_encoders/`
- `qwen_image_vae.safetensors` in `models/vae/`
- `RealESRGAN_x2.pth` in `models/upscale_models/`

If `tools/link_qwen_bundle.py` is used, the Qwen bundle files are linked automatically. If the RealESRGAN checkpoint is stored elsewhere, pass it explicitly with `--realesrgan`.

## How to open

- `qwen_mrflow_workflow.json` opens as a normal editable canvas with `Ctrl+O`.
- `qwen_mrflow_api.json` is the API-style prompt export.
- The ComfyUI template browser uses `subgraphs/qwen_mrflow.json`.

## Editable controls

The canvas workflow exposes `Batch Size`, `Low-Res Steps`, `Refine Steps`, and `Refine Strength` as normal controls.
