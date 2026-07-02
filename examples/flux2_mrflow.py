import argparse
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from RealESRGAN import RealESRGAN


SETTINGS = {
    "base4b_12plus1": {
        "stage1_steps": 12,
        "refine_steps": 1,
        "refine_sigma": 0.10,
        "guidance_scale": 4.0,
    },
    "base9b_12plus1": {
        "stage1_steps": 12,
        "refine_steps": 1,
        "refine_sigma": 0.10,
        "guidance_scale": 4.0,
    },
    "4b_4plus1": {
        "stage1_steps": 4,
        "refine_steps": 1,
        "refine_sigma": 0.25,
        "guidance_scale": 1.0,
    },
    "9b_4plus1": {
        "stage1_steps": 4,
        "refine_steps": 1,
        "refine_sigma": 0.25,
        "guidance_scale": 1.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal FLUX.2 Klein SDEdit-style MrFlow demo.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--setting", choices=SETTINGS.keys(), default="base9b_12plus1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--realesrgan-x2", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--low-width", type=int, default=512)
    parser.add_argument("--low-height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="mrflow_outputs/flux2_klein")
    return parser.parse_args()


def encode_image_latents(pipe, image, height, width, generator, dtype, device):
    init_image = pipe.image_processor.preprocess(image, height=height, width=width, resize_mode="crop")
    init_image = init_image.to(device=device, dtype=dtype)
    image_latents = pipe._encode_vae_image(init_image, generator=generator)
    latent_ids = pipe._prepare_latent_ids(image_latents).to(device)
    return image_latents, latent_ids


@torch.no_grad()
def sdedit_refine(
    pipe,
    image,
    prompt,
    args,
    cfg,
    prompt_embeds=None,
    text_ids=None,
    negative_prompt_embeds=None,
    negative_text_ids=None,
):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)

    pipe._guidance_scale = cfg["guidance_scale"]
    pipe._attention_kwargs = None
    pipe._current_timestep = None
    pipe._interrupt = False

    if prompt_embeds is None or text_ids is None:
        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

    if pipe.do_classifier_free_guidance and (negative_prompt_embeds is None or negative_text_ids is None):
        negative_prompt_embeds, negative_text_ids = pipe.encode_prompt(
            prompt="",
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

    image_latents, latent_ids = encode_image_latents(
        pipe,
        image=image,
        height=args.height,
        width=args.width,
        generator=generator,
        dtype=pipe.vae.dtype,
        device=device,
    )

    refine_steps = cfg["refine_steps"]
    sigmas = [cfg["refine_sigma"]]
    packed_seq_len = pipe._pack_latents(image_latents).shape[1]
    mu = compute_empirical_mu(image_seq_len=packed_seq_len, num_steps=refine_steps)
    retrieve_timesteps(pipe.scheduler, refine_steps, device, sigmas=sigmas, mu=mu)
    timesteps = pipe.scheduler.timesteps
    actual_steps = len(timesteps)
    pipe.scheduler.set_begin_index(0)

    latent_timestep = timesteps[:1].repeat(1)
    noise = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=image_latents.dtype)
    latents = pipe.scheduler.scale_noise(image_latents, latent_timestep, noise)
    latents = pipe._pack_latents(latents)

    with pipe.progress_bar(total=actual_steps) as progress_bar:
        for timestep in timesteps:
            if pipe.interrupt:
                continue

            timestep_input = timestep.expand(latents.shape[0]).to(latents.dtype)
            latent_model_input = latents.to(pipe.transformer.dtype)
            with pipe.transformer.cache_context("cond"):
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep_input / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=pipe.attention_kwargs,
                    return_dict=False,
                )[0]

            if pipe.do_classifier_free_guidance:
                with pipe.transformer.cache_context("uncond"):
                    neg_noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep_input / 1000,
                        guidance=None,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=pipe.attention_kwargs,
                        return_dict=False,
                    )[0]
                noise_pred = neg_noise_pred + cfg["guidance_scale"] * (noise_pred - neg_noise_pred)

            latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            progress_bar.update()

    latents = pipe._unpack_latents_with_ids(latents, latent_ids)
    bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(
        latents.device, latents.dtype
    )
    latents = latents * bn_std + bn_mean
    latents = pipe._unpatchify_latents(latents).to(pipe.vae.dtype)
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0], actual_steps


def main():
    args = parse_args()
    cfg = dict(SETTINGS[args.setting])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    pipe._guidance_scale = cfg["guidance_scale"]
    pipe._attention_kwargs = None
    pipe._current_timestep = None
    pipe._interrupt = False

    prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=args.prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )

    negative_prompt_embeds = None
    negative_text_ids = None
    if pipe.do_classifier_free_guidance:
        negative_prompt_embeds, negative_text_ids = pipe.encode_prompt(
            prompt="",
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

    # Stage 1: generate the global layout quickly in the low-resolution latent space.
    generator = torch.Generator(device=device).manual_seed(args.seed)
    low = pipe(
        prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        height=args.low_height,
        width=args.low_width,
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["stage1_steps"],
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
    # Stage 4: run a short high-resolution refine pass to recover final details.
    refined, actual_refine_steps = sdedit_refine(
        pipe,
        upscaled,
        args.prompt,
        args,
        cfg,
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_text_ids=negative_text_ids,
    )

    stem = f"flux2_mrflow_{args.setting}_seed{args.seed}_{args.width}x{args.height}"
    low_path = out_dir / f"{stem}_stage1_low.png"
    upscaled_path = out_dir / f"{stem}_stage2_upscaled.png"
    refined_path = out_dir / f"{stem}_stage3_refined.png"

    low.save(low_path)
    upscaled.save(upscaled_path)
    refined.save(refined_path)
    print(f"Actual refine steps: {actual_refine_steps}")
    print(f"Saved: {refined_path}")


if __name__ == "__main__":
    main()
