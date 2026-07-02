#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def ensure_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src)


def resolve_default_realesrgan(bundle: Path, comfyui: Path) -> Path | None:
    candidates = [
        bundle / "RealESRGAN_x2.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def main():
    parser = argparse.ArgumentParser(description="Link a local Qwen-Image_ComfyUI bundle into ComfyUI model folders.")
    parser.add_argument("--bundle", required=True, help="Path to the Qwen-Image_ComfyUI bundle root.")
    parser.add_argument("--comfyui", required=True, help="Path to the ComfyUI root.")
    parser.add_argument("--realesrgan", default=None, help="Optional path to a RealESRGAN x2 checkpoint to link into upscale_models.")
    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()
    comfyui = Path(args.comfyui).resolve()

    links = [
        (
            bundle / "split_files" / "diffusion_models" / "qwen_image_bf16.safetensors",
            comfyui / "models" / "diffusion_models" / "qwen_image_bf16.safetensors",
        ),
        (
            bundle / "split_files" / "text_encoders" / "qwen_2.5_vl_7b.safetensors",
            comfyui / "models" / "text_encoders" / "qwen_2.5_vl_7b.safetensors",
        ),
        (
            bundle / "split_files" / "vae" / "qwen_image_vae.safetensors",
            comfyui / "models" / "vae" / "qwen_image_vae.safetensors",
        ),
    ]

    for src, dst in links:
        if not src.exists():
            raise FileNotFoundError(f"Missing source file: {src}")
        ensure_symlink(src, dst)
        print(f"linked {dst} -> {src}")

    realesrgan = Path(args.realesrgan).resolve() if args.realesrgan else resolve_default_realesrgan(bundle, comfyui)
    if realesrgan is not None:
        if not realesrgan.exists():
            raise FileNotFoundError(f"Missing RealESRGAN checkpoint: {realesrgan}")
        dst = comfyui / "models" / "upscale_models" / realesrgan.name
        ensure_symlink(realesrgan, dst)
        print(f"linked {dst} -> {realesrgan}")
    else:
        print("warning: no RealESRGAN_x2.pth found automatically; pass --realesrgan to link one manually")


if __name__ == "__main__":
    main()
