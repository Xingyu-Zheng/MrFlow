from __future__ import annotations

import math
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview
import node_helpers
import torch
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from comfy_api.latest import ComfyExtension, io, ui
from nodes import MAX_RESOLUTION
from typing_extensions import override


def _resize_image(image: torch.Tensor, width: int, height: int, method: str = "bicubic") -> torch.Tensor:
    samples = image.movedim(-1, 1)
    resized = comfy.utils.common_upscale(samples, width, height, method, "disabled")
    return resized.movedim(1, -1)


def _flowmatch_shift(t: torch.Tensor, mu: float, sigma: float = 1.0) -> torch.Tensor:
    exp_mu = math.exp(mu)
    return exp_mu / (exp_mu + (1.0 / t - 1.0) ** sigma)


def _direct_sigma_shift_mu(steps: int) -> float:
    if steps <= 1:
        return 0.0
    return 0.25 * float(steps - 1)


def _direct_sigma_nodes(first_sigma: float, steps: int, device: torch.device) -> torch.Tensor:
    if not 0.0 < first_sigma < 1.0:
        raise ValueError(f"first_sigma must be in (0, 1), got {first_sigma}")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if steps == 1:
        return torch.tensor([float(first_sigma), 0.0], dtype=torch.float32, device=device)

    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32, device=device)
    mu = _direct_sigma_shift_mu(steps)
    shifted = _flowmatch_shift(base.clamp(1.0e-6, 1.0 - 1.0e-6), mu=mu)
    shifted = shifted - shifted[-1]
    shifted = shifted / shifted[0]
    shifted = shifted * float(first_sigma)
    shifted[0] = float(first_sigma)
    shifted[-1] = 0.0
    return shifted


class MrFlowQwenPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 16}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 16}),
                "preset": (["12plus1", "20plus1"], {"default": "12plus1"}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = (
        "low_width",
        "low_height",
        "target_width",
        "target_height",
        "refine_denoise",
        "stage1_steps",
        "refine_steps",
    )
    FUNCTION = "build"
    CATEGORY = "MrFlow/Qwen"

    PRESETS = {
        "12plus1": {"denoise": 0.12, "stage1_steps": 12, "refine_steps": 1},
        "20plus1": {"denoise": 0.15, "stage1_steps": 20, "refine_steps": 1},
    }

    def build(self, target_width: int, target_height: int, preset: str, upscale_factor: float):
        low_width = max(16, int(round(target_width / upscale_factor / 16.0)) * 16)
        low_height = max(16, int(round(target_height / upscale_factor / 16.0)) * 16)
        cfg = self.PRESETS[preset]
        return (
            low_width,
            low_height,
            target_width,
            target_height,
            cfg["denoise"],
            cfg["stage1_steps"],
            cfg["refine_steps"],
        )


class MrFlowUpscaleEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 16}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 16}),
                "resize_method": (["bicubic", "bilinear", "area", "nearest-exact"], {"default": "bicubic"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("upscaled_image", "prepared_latent")
    FUNCTION = "prepare"
    CATEGORY = "MrFlow/Qwen"

    def prepare(self, image, vae, upscale_model, target_width: int, target_height: int, resize_method: str):
        # Stage 2: upsample the decoded low-resolution image in pixel space with a dedicated SR model.
        upscaled = ImageUpscaleWithModel.execute(upscale_model, image)[0]
        if upscaled.shape[2] != target_width or upscaled.shape[1] != target_height:
            upscaled = _resize_image(upscaled, target_width, target_height, method=resize_method)
        # Stage 3: re-encode the super-resolved image so the high-resolution sampler can inject noise and refine it.
        if upscaled.shape[0] > 1:
            latent_batches = []
            for i in range(upscaled.shape[0]):
                encoded = vae.encode(upscaled[i:i + 1])
                latent_batches.append(encoded)
            encoded_latent = torch.cat(latent_batches, dim=0)
        else:
            encoded_latent = vae.encode(upscaled)
        latent = {"samples": encoded_latent}
        return (upscaled, latent)


class MrFlowAttachReferenceLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "attach"
    CATEGORY = "MrFlow/Qwen"

    def attach(self, positive, negative, latent):
        ref = latent["samples"]
        # Stage 3: attach the re-encoded high-resolution latent as the refinement reference for both CFG branches.
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [ref]}, append=True)
        return (positive, negative)


class MrFlowQwenRefine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 2026, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "denoise": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "print_schedule": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("refined_latent", "refined_image")
    FUNCTION = "refine"
    CATEGORY = "MrFlow/Qwen"

    def refine(self, model, vae, positive, negative, latent_image, seed, steps, cfg, sampler_name, denoise, print_schedule):
        # Stage 4: inject the exact first-step noise used by MrFlow and denoise with an explicit sigma path.
        latent = latent_image.copy()
        latent_samples = latent["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(
            model,
            latent_samples,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        latent["samples"] = latent_samples

        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        noise_mask = latent.get("noise_mask", None)
        sigmas = _direct_sigma_nodes(denoise, steps, device=model.load_device)
        sampler = comfy.samplers.sampler_object(sampler_name)
        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        refined_samples = comfy.sample.sample_custom(
            model,
            noise,
            cfg,
            sampler,
            sigmas,
            positive,
            negative,
            latent_samples,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
        refined_latent = latent.copy()
        refined_latent.pop("downscale_ratio_spacial", None)
        refined_latent.pop("downscale_ratio_temporal", None)
        refined_latent["samples"] = refined_samples
        refined_image = vae.decode(refined_latent["samples"])
        if len(refined_image.shape) == 5:
            refined_image = refined_image.reshape(-1, refined_image.shape[-3], refined_image.shape[-2], refined_image.shape[-1])
        return (refined_latent, refined_image)


class MrFlowSaveImage:
    def __init__(self):
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "MrFlow/output"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "MrFlow/Qwen"

    def save_images(self, images, filename_prefix="MrFlow/output", prompt=None, extra_pnginfo=None):
        saved = ui.ImageSaveHelper.get_save_images_ui(
            images,
            filename_prefix=filename_prefix,
            cls=None,
            compress_level=self.compress_level,
        )
        return {"ui": saved.as_dict(), "result": (images,)}


NODE_CLASS_MAPPINGS = {
    "MrFlowQwenPreset": MrFlowQwenPreset,
    "MrFlowUpscaleEncode": MrFlowUpscaleEncode,
    "MrFlowAttachReferenceLatent": MrFlowAttachReferenceLatent,
    "MrFlowQwenRefine": MrFlowQwenRefine,
    "MrFlowSaveImage": MrFlowSaveImage,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "MrFlowQwenPreset": "MrFlow Qwen Preset",
    "MrFlowUpscaleEncode": "MrFlow Upscale + Encode",
    "MrFlowAttachReferenceLatent": "MrFlow Attach Reference Latent",
    "MrFlowQwenRefine": "MrFlow Qwen Refine",
    "MrFlowSaveImage": "MrFlow Save Image",
}


class MrFlowQwenPresetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MrFlowQwenPreset",
            display_name="MrFlow Qwen Preset",
            category="MrFlow/Qwen",
            inputs=[
                io.Int.Input("target_width", default=1024, min=64, max=MAX_RESOLUTION, step=16),
                io.Int.Input("target_height", default=1024, min=64, max=MAX_RESOLUTION, step=16),
                io.Combo.Input("preset", options=["12plus1", "20plus1"], default="12plus1"),
                io.Float.Input("upscale_factor", default=2.0, min=1.0, max=8.0, step=0.05),
            ],
            outputs=[
                io.Int.Output(display_name="low_width"),
                io.Int.Output(display_name="low_height"),
                io.Int.Output(display_name="target_width"),
                io.Int.Output(display_name="target_height"),
                io.Float.Output(display_name="refine_denoise"),
                io.Int.Output(display_name="stage1_steps"),
                io.Int.Output(display_name="refine_steps"),
            ],
        )

    @classmethod
    def execute(cls, target_width, target_height, preset, upscale_factor):
        return io.NodeOutput(*MrFlowQwenPreset().build(target_width, target_height, preset, upscale_factor))


class MrFlowUpscaleEncodeNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MrFlowUpscaleEncode",
            display_name="MrFlow Upscale + Encode",
            category="MrFlow/Qwen",
            inputs=[
                io.Image.Input("image"),
                io.Vae.Input("vae"),
                io.UpscaleModel.Input("upscale_model"),
                io.Int.Input("target_width", default=1024, min=64, max=MAX_RESOLUTION, step=16),
                io.Int.Input("target_height", default=1024, min=64, max=MAX_RESOLUTION, step=16),
                io.Combo.Input("resize_method", options=["bicubic", "bilinear", "area", "nearest-exact"], default="bicubic"),
            ],
            outputs=[
                io.Image.Output(display_name="upscaled_image"),
                io.Latent.Output(display_name="prepared_latent"),
            ],
        )

    @classmethod
    def execute(cls, image, vae, upscale_model, target_width, target_height, resize_method):
        return io.NodeOutput(*MrFlowUpscaleEncode().prepare(image, vae, upscale_model, target_width, target_height, resize_method))


class MrFlowAttachReferenceLatentNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MrFlowAttachReferenceLatent",
            display_name="MrFlow Attach Reference Latent",
            category="MrFlow/Qwen",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, latent):
        return io.NodeOutput(*MrFlowAttachReferenceLatent().attach(positive, negative, latent))


class MrFlowQwenRefineNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MrFlowQwenRefine",
            display_name="MrFlow Qwen Refine",
            category="MrFlow/Qwen",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Int.Input("seed", default=2026, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Int.Input("steps", default=1, min=1, max=10000),
                io.Float.Input("cfg", default=4.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default=comfy.samplers.KSampler.SAMPLERS[0]),
                io.Float.Input("denoise", default=0.12, min=0.0, max=1.0, step=0.01, round=0.001),
                io.Boolean.Input("print_schedule", default=False),
            ],
            outputs=[
                io.Latent.Output(display_name="refined_latent"),
                io.Image.Output(display_name="refined_image"),
            ],
        )

    @classmethod
    def execute(cls, model, vae, positive, negative, latent_image, seed, steps, cfg, sampler_name, denoise, print_schedule):
        return io.NodeOutput(*MrFlowQwenRefine().refine(model, vae, positive, negative, latent_image, seed, steps, cfg, sampler_name, denoise, print_schedule))


class MrFlowSaveImageNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MrFlowSaveImage",
            display_name="MrFlow Save Image",
            category="MrFlow/Qwen",
            inputs=[
                io.Image.Input("images"),
                io.String.Input("filename_prefix", default="MrFlow/output"),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, filename_prefix, prompt=None, extra_pnginfo=None):
        result = MrFlowSaveImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return io.NodeOutput(result["result"][0], ui=result["ui"])


class MrFlowExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            MrFlowQwenPresetNode,
            MrFlowUpscaleEncodeNode,
            MrFlowAttachReferenceLatentNode,
            MrFlowQwenRefineNode,
            MrFlowSaveImageNode,
        ]


async def comfy_entrypoint() -> MrFlowExtension:
    return MrFlowExtension()
