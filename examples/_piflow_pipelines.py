import sys
from pathlib import Path

import torch
from diffusers.pipelines.flux.pipeline_flux_img2img import retrieve_latents as retrieve_flux_latents
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage_img2img import retrieve_latents as retrieve_qwen_latents
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = Path(__file__).resolve().parent
for path in (ROOT, EXAMPLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from piflow_local import adapter_policy, import_piflow_components, make_flow_map_scheduler  # noqa: E402


logger = logging.get_logger(__name__)


def _infer_batch_size(prompt, prompt_embeds):
    if isinstance(prompt, str):
        return 1
    if prompt is not None:
        return len(prompt)
    if prompt_embeds is None:
        raise ValueError("Pass either `prompt` or `prompt_embeds`.")
    return prompt_embeds.shape[0]


class PiQwenImageImg2ImgPipeline(import_piflow_components()[2]):
    def check_img2img_inputs(self, prompt, strength, height, width, prompt_embeds=None, max_sequence_length=None):
        if not 0 <= strength <= 1:
            raise ValueError(f"`strength` must be in [0, 1], got {strength}.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Pass either `prompt` or `prompt_embeds`, not both.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Pass either `prompt` or `prompt_embeds`.")
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` must be str or list[str], got {type(prompt)}.")
        if max_sequence_length is not None and max_sequence_length > 1024:
            raise ValueError(f"`max_sequence_length` cannot exceed 1024, got {max_sequence_length}.")

    def _encode_vae_image(self, image, generator):
        image_latents = retrieve_qwen_latents(self.vae.encode(image), generator=generator)
        mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            image_latents.device, image_latents.dtype
        )
        std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            image_latents.device, image_latents.dtype
        )
        return (image_latents - mean) / std

    def prepare_img2img_latents(self, image, timestep, batch_size, num_channels_latents, height, width, dtype, device, generator):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, 1, num_channels_latents, height, width)
        if image.dim() == 4:
            image = image.unsqueeze(2)
        image = image.to(device=device, dtype=dtype)
        image_latents = self._encode_vae_image(image, generator=generator)
        image_latents = image_latents.transpose(1, 2)
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        sigma = (timestep / self.scheduler.config.num_train_timesteps).to(device=device, dtype=dtype)
        while sigma.dim() < image_latents.dim():
            sigma = sigma.view(-1, *([1] * (image_latents.dim() - 1)))
        latents = (1 - sigma) * image_latents + sigma * noise
        return self._pack_latents(latents, batch_size, num_channels_latents, height, width)

    def set_img2img_timesteps(self, num_inference_steps, strength, seq_len, device):
        self.scheduler.set_timesteps(num_inference_steps, seq_len=seq_len, device=device)
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        self.scheduler.timesteps = self.scheduler.timesteps[t_start:]
        self.scheduler.timesteps_dst = self.scheduler.timesteps_dst[t_start:]
        self.scheduler.sigmas = self.scheduler.sigmas[t_start:]
        self.scheduler.m_vals = self.scheduler.m_vals[t_start:]
        self.scheduler.set_begin_index(0)
        self.scheduler._step_index = None
        return self.scheduler.timesteps, self.scheduler.timesteps_dst

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        image=None,
        height=None,
        width=None,
        strength=0.25,
        num_inference_steps=4,
        total_substeps=128,
        temperature="auto",
        generator=None,
        max_sequence_length=512,
        output_type="pil",
        return_dict=True,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        attention_kwargs=None,
        **kwargs,
    ):
        if image is None:
            return super().__call__(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                total_substeps=total_substeps,
                temperature=temperature,
                generator=generator,
                max_sequence_length=max_sequence_length,
                output_type=output_type,
                return_dict=return_dict,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                attention_kwargs=attention_kwargs,
                **kwargs,
            )

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self.check_img2img_inputs(prompt, strength, height, width, prompt_embeds, max_sequence_length)
        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        init_image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        batch_size = _infer_batch_size(prompt, prompt_embeds)
        device = self._execution_device
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        timesteps, timesteps_dst = self.set_img2img_timesteps(num_inference_steps, strength, image_seq_len, device)
        latent_timestep = timesteps[:1].repeat(batch_size)
        latents = self.prepare_img2img_latents(
            init_image,
            latent_timestep,
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )
        img_shapes = [[(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]] * batch_size

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for index, (t_src, t_dst) in enumerate(zip(timesteps, timesteps_dst)):
                self._current_timestep = t_src
                sigma_t_src = t_src / self.scheduler.config.num_train_timesteps
                sigma_t_dst = t_dst / self.scheduler.config.num_train_timesteps
                with self.transformer.cache_context("cond"):
                    denoising_output = self.transformer(
                        hidden_states=latents.to(dtype=self.transformer.dtype),
                        timestep=t_src.expand(latents.shape[0]) / 1000,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        attention_kwargs=self.attention_kwargs,
                    )
                latents = self._unpack_latents(latents, height, width, self.vae_scale_factor, target_patch_size=1)
                denoising_output = self._unpack_gm(denoising_output, height, width, num_channels_latents, gm_patch_size=1)
                denoising_output = {k: v.to(torch.float32) for k, v in denoising_output.items()}
                policy = self.policy_class(denoising_output, latents, sigma_t_src)
                if index < len(timesteps) - 1:
                    policy.temperature_(min(max(0.1 * (num_inference_steps - 1), 0), 1) if temperature == "auto" else float(temperature))
                batch = latents.size(0)
                sigma_src = sigma_t_src.reshape(1).expand(batch)
                sigma_dst = sigma_t_dst.reshape(1).expand(batch)
                raw_src = self.scheduler.unwarp_t(sigma_src, seq_len=image_seq_len)
                raw_dst = self.scheduler.unwarp_t(sigma_dst, seq_len=image_seq_len)
                latents_dst, _, _ = policy.integrate(
                    latents,
                    sigma_src,
                    raw_src,
                    raw_dst,
                    self.scheduler,
                    seq_len=image_seq_len,
                    total_substeps=total_substeps,
                )
                latents = self.scheduler.step(latents_dst, t_src, latents, return_dict=False)[0]
                latents = self._pack_latents(
                    latents,
                    batch,
                    num_channels_latents,
                    2 * (int(height) // (self.vae_scale_factor * 2)),
                    2 * (int(width) // (self.vae_scale_factor * 2)),
                    patch_size=1,
                )
                progress_bar.update()

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)[:, :, None]
        mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * std + mean
        image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0][:, :, 0]
        image = self.image_processor.postprocess(image, output_type=output_type)
        return QwenImagePipelineOutput(images=image) if return_dict else (image,)


class PiFluxImg2ImgPipeline(import_piflow_components()[1]):
    def _encode_vae_image(self, image, generator):
        image_latents = retrieve_flux_latents(self.vae.encode(image), generator=generator)
        return (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    def prepare_img2img_latents(self, image, timestep, batch_size, num_channels_latents, height, width, dtype, device, generator):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
        image = image.to(device=device, dtype=dtype)
        image_latents = self._encode_vae_image(image=image, generator=generator)
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        sigma = (timestep / self.scheduler.config.num_train_timesteps).to(device=device, dtype=dtype)
        while sigma.dim() < image_latents.dim():
            sigma = sigma.view(-1, *([1] * (image_latents.dim() - 1)))
        latents = (1 - sigma) * image_latents + sigma * noise
        return self._pack_latents(latents, batch_size, num_channels_latents, height, width), latent_image_ids

    def set_img2img_timesteps(self, num_inference_steps, strength, seq_len, device):
        self.scheduler.set_timesteps(num_inference_steps, seq_len=seq_len, device=device)
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        self.scheduler.timesteps = self.scheduler.timesteps[t_start:]
        self.scheduler.timesteps_dst = self.scheduler.timesteps_dst[t_start:]
        self.scheduler.sigmas = self.scheduler.sigmas[t_start:]
        self.scheduler.m_vals = self.scheduler.m_vals[t_start:]
        self.scheduler.set_begin_index(0)
        self.scheduler._step_index = None
        return self.scheduler.timesteps, self.scheduler.timesteps_dst

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        prompt_2=None,
        image=None,
        height=None,
        width=None,
        strength=0.25,
        num_inference_steps=4,
        total_substeps=128,
        temperature="auto",
        guidance_scale=3.5,
        generator=None,
        max_sequence_length=512,
        output_type="pil",
        return_dict=True,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        joint_attention_kwargs=None,
        **kwargs,
    ):
        if image is None:
            return super().__call__(
                prompt=prompt,
                prompt_2=prompt_2,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                total_substeps=total_substeps,
                temperature=temperature,
                guidance_scale=guidance_scale,
                generator=generator,
                max_sequence_length=max_sequence_length,
                output_type=output_type,
                return_dict=return_dict,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                joint_attention_kwargs=joint_attention_kwargs,
                **kwargs,
            )

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        init_image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        batch_size = _infer_batch_size(prompt, prompt_embeds)
        device = self._execution_device
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        timesteps, timesteps_dst = self.set_img2img_timesteps(num_inference_steps, strength, image_seq_len, device)
        latent_timestep = timesteps[:1].repeat(batch_size)
        latents, latent_image_ids = self.prepare_img2img_latents(
            init_image,
            latent_timestep,
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0])
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for index, (t_src, t_dst) in enumerate(zip(timesteps, timesteps_dst)):
                sigma_t_src = t_src / self.scheduler.config.num_train_timesteps
                sigma_t_dst = t_dst / self.scheduler.config.num_train_timesteps
                with self.transformer.cache_context("cond"):
                    denoising_output = self.transformer(
                        hidden_states=latents.to(dtype=self.transformer.dtype),
                        timestep=t_src.expand(latents.shape[0]) / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                    )
                latents = self._unpack_latents(latents, height, width, self.vae_scale_factor, target_patch_size=1)
                denoising_output = self._unpack_gm(denoising_output, height, width, num_channels_latents, gm_patch_size=1)
                denoising_output = {k: v.to(torch.float32) for k, v in denoising_output.items()}
                policy = self.policy_class(denoising_output, latents, sigma_t_src)
                if index < len(timesteps) - 1:
                    policy.temperature_(min(max(0.1 * (num_inference_steps - 1), 0), 1) if temperature == "auto" else float(temperature))
                batch = latents.size(0)
                sigma_src = sigma_t_src.reshape(1).expand(batch)
                sigma_dst = sigma_t_dst.reshape(1).expand(batch)
                raw_src = self.scheduler.unwarp_t(sigma_src, seq_len=image_seq_len)
                raw_dst = self.scheduler.unwarp_t(sigma_dst, seq_len=image_seq_len)
                latents_dst, _, _ = policy.integrate(
                    latents,
                    sigma_src,
                    raw_src,
                    raw_dst,
                    self.scheduler,
                    seq_len=image_seq_len,
                    total_substeps=total_substeps,
                )
                latents = self.scheduler.step(latents_dst, t_src, latents, return_dict=False)[0]
                latents = self._pack_latents(
                    latents,
                    batch,
                    num_channels_latents,
                    2 * (int(height) // (self.vae_scale_factor * 2)),
                    2 * (int(width) // (self.vae_scale_factor * 2)),
                    patch_size=1,
                )
                progress_bar.update()

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
        return FluxPipelineOutput(images=image) if return_dict else (image,)


def load_piflow_pipe(pipeline_cls, model, adapter_root, adapter, dtype, shift, final_step_size_scale):
    policy_type, policy_kwargs = adapter_policy(adapter)
    pipe = pipeline_cls.from_pretrained(model, torch_dtype=dtype, policy_type=policy_type, policy_kwargs=policy_kwargs)
    pipe.load_lakonlab_adapter(adapter_root, subfolder=adapter, target_module_name="transformer", local_files_only=True)
    pipe.scheduler = make_flow_map_scheduler(pipe, shift=shift, final_step_size_scale=final_step_size_scale)
    return pipe
