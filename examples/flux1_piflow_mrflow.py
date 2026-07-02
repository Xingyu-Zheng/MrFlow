import argparse
import sys
from pathlib import Path

import torch
from RealESRGAN import RealESRGAN


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _piflow_pipelines import PiFluxImg2ImgPipeline, load_piflow_pipe  # noqa: E402
from piflow_local import make_flow_map_scheduler  # noqa: E402


SETTING = {
    "stage1_steps": 4,
    "refine_steps": 4,
    "strength": 0.25,
    "adapter": "gmflux_k8_piid_4step",
    "shift": 3.2,
    "stage1_final_step_size_scale": 0.5,
    "stage3_final_step_size_scale": 1.0,
    "total_substeps": 128,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal FLUX.1-dev PiFlow + MrFlow demo.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-root", required=True)
    parser.add_argument("--realesrgan-x2", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--low-width", type=int, default=512)
    parser.add_argument("--low-height", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="mrflow_outputs/flux1_piflow")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = SETTING
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    pipe = load_piflow_pipe(
        PiFluxImg2ImgPipeline,
        model=args.model,
        adapter_root=args.adapter_root,
        adapter=cfg["adapter"],
        dtype=dtype,
        shift=cfg["shift"],
        final_step_size_scale=cfg["stage1_final_step_size_scale"],
    ).to(device)
    # Stage 1: generate the global layout quickly in the low-resolution latent space.
    prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=args.prompt,
        device=device,
        num_images_per_prompt=1,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    low = pipe(
        prompt=None,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        width=args.low_width,
        height=args.low_height,
        num_inference_steps=cfg["stage1_steps"],
        total_substeps=cfg["total_substeps"],
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    # Stage 2: upsample the decoded low-resolution image in pixel space with x2 super-resolution.
    sr = RealESRGAN(device, scale=2)
    sr.load_weights(args.realesrgan_x2, download=False)
    upscaled = sr.predict(low)
    if upscaled.size != (args.width, args.height):
        upscaled = upscaled.resize((args.width, args.height))
    del sr
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Stage 3: inject a small amount of matched noise after re-encoding the super-resolved image.
    pipe.scheduler = make_flow_map_scheduler(
        pipe,
        shift=cfg["shift"],
        final_step_size_scale=cfg["stage3_final_step_size_scale"],
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    # Stage 4: run a short high-resolution refine pass to recover final details.
    refined = pipe(
        prompt=None,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        image=upscaled,
        width=args.width,
        height=args.height,
        strength=cfg["strength"],
        num_inference_steps=cfg["refine_steps"],
        total_substeps=cfg["total_substeps"],
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    stem = f"flux1_piflow_mrflow_4plus1_seed{args.seed}_{args.width}x{args.height}"
    low.save(out_dir / f"{stem}_stage1_low.png")
    upscaled.save(out_dir / f"{stem}_stage2_upscaled.png")
    refined_path = out_dir / f"{stem}_stage3_refined.png"
    refined.save(refined_path)
    print(f"Saved: {refined_path}")


if __name__ == "__main__":
    main()
