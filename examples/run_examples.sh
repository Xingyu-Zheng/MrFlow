#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROMPT="Close-up of a chubby golden British shorthair on a knitted blanket, tiny ears, plush cheeks, bright cozy bedroom light."
REALESRGAN_X2="/path/to/RealESRGAN_x2.pth"
OUT_ROOT="mrflow_outputs"
SEED=2026

QWEN_IMAGE="/path/to/Qwen-Image"
FLUX1_DEV="/path/to/FLUX.1-dev"
FLUX2_KLEIN_BASE_9B="/path/to/FLUX.2-klein-base-9B"
FLUX2_KLEIN_9B="/path/to/FLUX.2-klein-9B"
Z_IMAGE_TURBO="/path/to/Z-Image-Turbo"

PI_QWEN_ADAPTER_ROOT="/path/to/pi-Qwen-Image"
PI_FLUX_ADAPTER_ROOT="/path/to/pi-FLUX.1"

mkdir -p "${OUT_ROOT}"

python "${SCRIPT_DIR}/qwen_image_mrflow.py" \
  --prompt "${PROMPT}" \
  --model "${QWEN_IMAGE}" \
  --realesrgan-x2 "${REALESRGAN_X2}" \
  --setting 12plus1 \
  --seed "${SEED}" \
  --output-dir "${OUT_ROOT}/qwen_image_mrflow_12plus1"

python "${SCRIPT_DIR}/flux1_mrflow.py" \
  --prompt "${PROMPT}" \
  --model "${FLUX1_DEV}" \
  --realesrgan-x2 "${REALESRGAN_X2}" \
  --setting 12plus1 \
  --seed "${SEED}" \
  --output-dir "${OUT_ROOT}/flux1_mrflow_12plus1"

python "${SCRIPT_DIR}/qwen_image_piflow_mrflow.py" \
  --prompt "${PROMPT}" \
  --model "${QWEN_IMAGE}" \
  --adapter-root "${PI_QWEN_ADAPTER_ROOT}" \
  --realesrgan-x2 "${REALESRGAN_X2}" \
  --seed "${SEED}" \
  --output-dir "${OUT_ROOT}/qwen_image_piflow_mrflow_4plus1"

python "${SCRIPT_DIR}/flux1_piflow_mrflow.py" \
  --prompt "${PROMPT}" \
  --model "${FLUX1_DEV}" \
  --adapter-root "${PI_FLUX_ADAPTER_ROOT}" \
  --realesrgan-x2 "${REALESRGAN_X2}" \
  --seed "${SEED}" \
  --output-dir "${OUT_ROOT}/flux1_piflow_mrflow_4plus1"

python "${SCRIPT_DIR}/flux2_mrflow.py" \
  --prompt "${PROMPT}" \
  --model "${FLUX2_KLEIN_BASE_9B}" \
  --realesrgan-x2 "${REALESRGAN_X2}" \
  --setting base9b_12plus1 \
  --seed "${SEED}" \
  --output-dir "${OUT_ROOT}/flux2_base9b_mrflow_12plus1"

python "${SCRIPT_DIR}/flux2_mrflow.py" \
  --prompt "${PROMPT}" \
  --model "${FLUX2_KLEIN_9B}" \
  --realesrgan-x2 "${REALESRGAN_X2}" \
  --setting 9b_4plus1 \
  --seed "${SEED}" \
  --output-dir "${OUT_ROOT}/flux2_9b_mrflow_4plus1"

python "${SCRIPT_DIR}/zimage_turbo_mrflow.py" \
  --prompt "${PROMPT}" \
  --model "${Z_IMAGE_TURBO}" \
  --realesrgan-x2 "${REALESRGAN_X2}" \
  --seed "${SEED}" \
  --output-dir "${OUT_ROOT}/zimage_turbo_mrflow_9plus1"
