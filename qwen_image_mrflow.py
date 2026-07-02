from pathlib import Path
import sys

import torch
from diffusers import QwenImageImg2ImgPipeline, QwenImagePipeline
from RealESRGAN import RealESRGAN

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrflow_utils import direct_sigma_schedule


PROMPT = "Close-up of a chubby golden British shorthair on a knitted blanket, tiny ears, plush cheeks, bright cozy bedroom light."
NEGATIVE_PROMPT = " "
MODEL = "/path/to/Qwen-Image"
REALESRGAN_X2 = "/path/to/RealESRGAN_x2.pth"
OUT_DIR = Path("outputs/qwen_image_mrflow_12plus1")

SEED = 2026
DEVICE = "cuda"
LOW_SIZE = 512
HIGH_SIZE = 1024
LOW_STEPS = 12
REFINE_STEPS = 1
REFINE_SIGMA = 0.12
TRUE_CFG_SCALE = 4.0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Stage 1: generate the global layout quickly in the low-resolution latent space.
    pipe = QwenImagePipeline.from_pretrained(MODEL, torch_dtype=dtype).to(device)
    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
        PROMPT,
        device=device,
        num_images_per_prompt=1,
    )
    negative_prompt_embeds, negative_prompt_embeds_mask = pipe.encode_prompt(
        NEGATIVE_PROMPT,
        device=device,
        num_images_per_prompt=1,
    )
    generator = torch.Generator(device=device).manual_seed(SEED)
    low = pipe(
        prompt=None,
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        width=LOW_SIZE,
        height=LOW_SIZE,
        num_inference_steps=LOW_STEPS,
        true_cfg_scale=TRUE_CFG_SCALE,
        generator=generator,
    ).images[0]

    # Stage 2: upsample the decoded low-resolution image in pixel space with x2 super-resolution.
    sr = RealESRGAN(device, scale=2)
    sr.load_weights(REALESRGAN_X2, download=False)
    upscaled = sr.predict(low).resize((HIGH_SIZE, HIGH_SIZE))

    # Stage 3: inject a small amount of matched noise after re-encoding the super-resolved image.
    refiner = QwenImageImg2ImgPipeline(**pipe.components).to(device)
    generator = torch.Generator(device=device).manual_seed(SEED + 1)
    with direct_sigma_schedule(refiner.scheduler, REFINE_SIGMA, REFINE_STEPS, device):
        # Stage 4: run a short high-resolution refine pass to recover final details.
        refined = refiner(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            image=upscaled,
            width=HIGH_SIZE,
            height=HIGH_SIZE,
            strength=1.0,
            num_inference_steps=REFINE_STEPS,
            true_cfg_scale=TRUE_CFG_SCALE,
            generator=generator,
        ).images[0]

    low.save(OUT_DIR / "stage1_low.png")
    upscaled.save(OUT_DIR / "stage2_upscaled.png")
    refined.save(OUT_DIR / "stage3_refined.png")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
