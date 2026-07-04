

# ComfyUI-Rebels-MrFlow

**ZIT Mr. Flow** and **Krea-2 Mr. Flow** — MrFlow training-free staged sampling
(low-res generate → pixel SR upscale → re-encode → 1-step refine) ported to
Z-Image Turbo and Krea-2 for ComfyUI.

Method credit: [MrFlow by Xingyu-Zheng et al.](https://github.com/Xingyu-Zheng/MrFlow)
(arXiv:2607.01642). Port by RealRebelAI. Works with GGUF / NF4 / FP8 / safetensors
loaders — anything that outputs a normal MODEL. No extra dependencies.

# Examples
- Krea-2 (1024)















- Krea-2 (2048)





## Nodes

| Node | What it does |
| --- | --- |
| ZIT Mr. Flow Preset / Krea-2 Mr. Flow Preset | Outputs low-res dims + all stage numbers for your target resolution |
| Mr. Flow Upscale + Encode (Rebels) | Stage 2+3: SR-model pixel upscale, snap to target size, VAE re-encode |
| ZIT Mr. Flow Refine / Krea-2 Mr. Flow Refine | Stage 4: matched noise injection + short explicit-sigma refine |

## Wiring (both models, identical graph)

1. **Preset node** → set `target_width/height`, pick preset.
2. **Empty Latent Image** ← `low_width` / `low_height` from preset.
3. **KSampler (stage 1)** — your normal model/CLIP/VAE loaders, steps ← `stage1_steps`,
   cfg ← `cfg`, sampler `euler`, scheduler `simple`, denoise 1.0.
4. **VAE Decode** stage-1 latent.
5. **Load Upscale Model** — RealESRGAN x2 for 1024 → **Mr. Flow Upscale + Encode**
   with the decoded image, your VAE, and `target_width/height` from preset.
6. **Refine node** — same model + conditioning as stage 1, `prepared_latent` in,
   steps ← `refine_steps`, denoise ← `refine_denoise`, cfg ← `cfg`, sampler `euler`.
7. **Save Image** from `refined_image`.

## Presets

- **ZIT `9plus1 (paper)`** — exact official MrFlow Z-Image Turbo demo numbers:
  9 low-res steps + 1 refine step at denoise 0.11, cfg 1.0 (no CFG).
- **Krea-2 `base_12plus1` / `base_20plus1`** — MrFlow's full-CFG regime (Qwen-style),
  cfg 4.0. Starting points — tune denoise 0.10–0.16 to taste.
- **Krea-2 `turbo_8plus1`** — for Krea-2 Turbo, cfg 1.0.

`schedule` on the refine node: `linear` (default, matches the official Z-Image
demo) or `shifted` (matches the official Qwen ComfyUI refine — only differs
when refine steps > 1).

## Notes

- Use a real SR model for stage 2 (RealESRGAN x2 for 1024)
  Plain latent upscaling defeats the whole method.
  
  - grab `RealESRGAN_x2plus.pth` from the [official releases page](https://huggingface.co/rklaumbach/RealESRGAN_x2/blob/main/RealESRGAN_x2.pth)
  

  and place it in `ComfyUI/models/upscale_models/` yourself.
  
- The Qwen `reference_latents` attach from upstream is intentionally omitted —
  it's a Qwen-Image-specific conditioning mechanism that Z-Image and Krea-2 don't use.
- Total budget: dominated by the cheap low-res pass; the refine is 1 step at
  full res. On 8GB VRAM this is a big win vs. sampling at target res directly.
