import argparse
import sys
from pathlib import Path

import torch
from RealESRGAN import RealESRGAN


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zimage_utils import load_zimage_pipeline, make_zimage_refiner, refine_zimage  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Z-Image-Turbo MrFlow demo.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--realesrgan-x2", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--low-width", type=int, default=512)
    parser.add_argument("--low-height", type=int, default=512)
    parser.add_argument("--stage1-steps", type=int, default=9)
    parser.add_argument("--refine-steps", type=int, default=9)
    parser.add_argument("--strength", type=float, default=0.11)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="mrflow_outputs/zimage_turbo")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    # Stage 1: generate the global layout quickly in the low-resolution latent space.
    pipe = load_zimage_pipeline(args.model, device=device, dtype=dtype)
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        device=pipe._execution_device,
        do_classifier_free_guidance=args.guidance_scale > 0,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    low = pipe(
        prompt=None,
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        height=args.low_height,
        width=args.low_width,
        num_inference_steps=args.stage1_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    # Stage 2: upsample the decoded low-resolution image in pixel space with x2 super-resolution.
    refiner = make_zimage_refiner(pipe, device=device)
    del pipe
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Stage 3: inject a small amount of matched noise after re-encoding the super-resolved image.
    sr = RealESRGAN(device, scale=2)
    sr.load_weights(args.realesrgan_x2, download=False)
    upscaled = sr.predict(low)
    if upscaled.size != (args.width, args.height):
        upscaled = upscaled.resize((args.width, args.height))
    del sr
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Stage 4: run a short high-resolution refine pass to recover final details.
    refined = refine_zimage(
        refiner,
        prompt=None,
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        image=upscaled,
        strength=args.strength,
        steps=args.refine_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed + 1,
    )

    stem = f"zimage_turbo_mrflow_9plus1_seed{args.seed}_{args.width}x{args.height}"
    low_path = out_dir / f"{stem}_stage1_low.png"
    upscaled_path = out_dir / f"{stem}_stage2_upscaled.png"
    refined_path = out_dir / f"{stem}_stage3_refined.png"

    low.save(low_path)
    upscaled.save(upscaled_path)
    refined.save(refined_path)
    print(f"Saved: {refined_path}")


if __name__ == "__main__":
    main()
