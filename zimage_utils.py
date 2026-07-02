import torch
from diffusers import ZImageImg2ImgPipeline, ZImagePipeline


def load_zimage_pipeline(model_path, device="cuda", dtype=torch.bfloat16):
    pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    return pipe.to(device)


def make_zimage_refiner(pipe, device="cuda"):
    return ZImageImg2ImgPipeline(**pipe.components).to(device)


def refine_zimage(
    refiner,
    prompt,
    image,
    strength,
    steps,
    guidance_scale,
    seed,
    negative_prompt="",
    prompt_embeds=None,
    negative_prompt_embeds=None,
    cfg_normalization=False,
):
    generator = torch.Generator(device=refiner._execution_device).manual_seed(seed)
    return refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        image=image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        cfg_normalization=cfg_normalization,
        generator=generator,
    ).images[0]
