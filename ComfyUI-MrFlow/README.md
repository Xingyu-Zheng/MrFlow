# ComfyUI-MrFlow

MrFlow helper nodes for ComfyUI Qwen workflows.

This extension adds a compact Qwen-specific path around the standard ComfyUI loaders, upscale model, and sampler nodes. It supports a four-stage flow:

1. low-resolution generation
2. pixel-space super-resolution
3. latent re-encode for refinement
4. high-resolution refine

## Included files

- `nodes_mrflow_qwen.py`: Qwen-oriented helper nodes
- `tools/link_qwen_bundle.py`: model-link helper for split Qwen bundles
- `examples/qwen_mrflow_workflow.json`: editable canvas workflow for `Ctrl+O`
- `examples/qwen_mrflow_api.json`: API-style prompt JSON
- `subgraphs/qwen_mrflow.json`: template-browser subgraph

## Installation

Place or symlink this folder in `ComfyUI/custom_nodes/`, then activate the ComfyUI Python environment:

```bash
conda activate comfyui
```

No extra Python packages are required beyond the ComfyUI environment.

## Model setup

If you have a split Qwen-Image bundle, link its files into ComfyUI with:

```bash
conda activate comfyui
python tools/link_qwen_bundle.py \
  --bundle /path/to/Qwen-Image_ComfyUI \
  --comfyui /path/to/ComfyUI
```

If the RealESRGAN x2 checkpoint is not inside the bundle directory, pass it explicitly:

```bash
python tools/link_qwen_bundle.py \
  --bundle /path/to/Qwen-Image_ComfyUI \
  --comfyui /path/to/ComfyUI \
  --realesrgan /path/to/RealESRGAN_x2.pth
```

## Usage

1. Restart ComfyUI after installing the extension.
2. Open `examples/qwen_mrflow_workflow.json` with `Ctrl+O`, or load `subgraphs/qwen_mrflow.json` from the template browser.
3. Replace the placeholder model names with files present in your local ComfyUI model folders.
4. Adjust `Batch Size`, `Low-Res Steps`, `Refine Steps`, and `Refine Strength` directly in the canvas.

## Notes

- The default values match the common `12+1` setting.
- `print_schedule` can be enabled on the refine node to show the sigma schedule in the terminal.
- The workflow stays compatible with normal ComfyUI save, sampler, and model-selection nodes.
