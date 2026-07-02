import argparse
from pathlib import Path

import torch
from diffusers import FluxImg2ImgPipeline, FluxPipeline
from RealESRGAN import RealESRGAN

from direct_sigma_refine import force_direct_sigma_set_timesteps


SETTINGS = {
    "20plus1": {"stage1_steps": 20, "refine_steps": 1, "refine_sigma": 0.15},
    "12plus1": {"stage1_steps": 12, "refine_steps": 1, "refine_sigma": 0.12},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal FLUX.1-dev MrFlow demo.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--realesrgan-x2", required=True)
    parser.add_argument("--setting", choices=SETTINGS.keys(), default="12plus1")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--low-width", type=int, default=512)
    parser.add_argument("--low-height", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="mrflow_outputs/flux1_dev")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = SETTINGS[args.setting]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Stage 1: generate the global layout quickly in the low-resolution latent space.
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
        args.prompt,
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
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    # Stage 2: upsample the decoded low-resolution image in pixel space with x2 super-resolution.
    refiner = FluxImg2ImgPipeline(**pipe.components).to(device)
    del pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Stage 3: inject a small amount of matched noise after re-encoding the super-resolved image.
    sr = RealESRGAN(device, scale=2)
    sr.load_weights(args.realesrgan_x2, download=False)
    upscaled = sr.predict(low)
    if upscaled.size != (args.width, args.height):
        upscaled = upscaled.resize((args.width, args.height))
    del sr
    if device.type == "cuda":
        torch.cuda.empty_cache()

    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    with force_direct_sigma_set_timesteps(refiner.scheduler, cfg["refine_sigma"], cfg["refine_steps"], device):
        # Stage 4: run a short high-resolution refine pass to recover final details.
        refined = refiner(
            prompt=None,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            image=upscaled,
            width=args.width,
            height=args.height,
            strength=1.0,
            num_inference_steps=cfg["refine_steps"],
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

    stem = f"flux1_mrflow_{args.setting}_seed{args.seed}_{args.width}x{args.height}"
    low_path = out_dir / f"{stem}_stage1_low.png"
    upscaled_path = out_dir / f"{stem}_stage2_upscaled.png"
    refined_path = out_dir / f"{stem}_stage3_refined.png"

    low.save(low_path)
    upscaled.save(upscaled_path)
    refined.save(refined_path)
    print(f"Saved: {refined_path}")


if __name__ == "__main__":
    main()
