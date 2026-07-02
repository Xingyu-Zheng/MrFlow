# MrFlow Example Scripts

This directory provides parameterized MrFlow examples for different model families and operating points. The scripts expose prompts, checkpoint paths, random seeds, output directories, and refinement settings through command-line arguments.

## Main Settings

| Setting | Low-resolution steps | Refinement steps | Refinement sigma | Use case |
| --- | ---: | ---: | ---: | --- |
| `12plus1` | 12 | 1 | `0.12` | Aggressive acceleration. |
| `20plus1` | 20 | 1 | `0.15` | Higher-quality operating point. |

The direct-sigma schedule explicitly specifies the starting noise level of the high-resolution refinement stage.

For `flux2_mrflow.py`, the available presets are:

| Setting | Variant | Low-resolution steps | Refinement steps | Refinement sigma | Guidance scale |
| --- | --- | ---: | ---: | ---: | ---: |
| `base4b_12plus1` | FLUX.2 Klein Base 4B | 12 | 1 | `0.10` | `4.0` |
| `base9b_12plus1` | FLUX.2 Klein Base 9B | 12 | 1 | `0.10` | `4.0` |
| `4b_4plus1` | FLUX.2 Klein 4B | 4 | 1 | `0.25` | `1.0` |
| `9b_4plus1` | FLUX.2 Klein 9B | 4 | 1 | `0.25` | `1.0` |

For `zimage_turbo_mrflow.py`, the default operating point uses `--stage1-steps 9`, `--refine-steps 9`, and `--strength 0.11`. These values are exposed as command-line arguments because Z-Image-Turbo uses its own reduced-step schedule.

## Script Index

| Script | Backbone | Notes |
| --- | --- | --- |
| `qwen_image_mrflow.py` | Qwen-Image | Training-free MrFlow. |
| `flux1_mrflow.py` | FLUX.1-dev | Training-free MrFlow. |
| `qwen_image_piflow_mrflow.py` | Qwen-Image + Pi-Flow | Uses distilled adapter/checkpoint inputs. |
| `flux1_piflow_mrflow.py` | FLUX.1-dev + Pi-Flow | Uses distilled adapter/checkpoint inputs. |
| `flux2_mrflow.py` | FLUX.2 Klein | Supports base and non-base settings. |
| `zimage_turbo_mrflow.py` | Z-Image-Turbo | Adds MrFlow refinement to a reduced-step model. |
| `direct_sigma_refine.py` | Shared helper | Builds explicit direct-sigma refinement schedules. |

## Usage

Qwen-Image:

```bash
python examples/qwen_image_mrflow.py \
  --prompt "${PROMPT}" \
  --model "${QWEN_IMAGE}" \
  --realesrgan-x2 "${REALESRGAN_X2}" \
  --setting 12plus1
```

FLUX.1-dev:

```bash
python examples/flux1_mrflow.py \
  --prompt "${PROMPT}" \
  --model "${FLUX1_DEV}" \
  --realesrgan-x2 "${REALESRGAN_X2}" \
  --setting 20plus1
```

Qwen-Image + Pi-Flow:

```bash
python examples/qwen_image_piflow_mrflow.py \
  --prompt "${PROMPT}" \
  --model "${QWEN_IMAGE}" \
  --adapter-root "${PI_QWEN_ADAPTER_ROOT}" \
  --realesrgan-x2 "${REALESRGAN_X2}"
```

FLUX.1-dev + Pi-Flow:

```bash
python examples/flux1_piflow_mrflow.py \
  --prompt "${PROMPT}" \
  --model "${FLUX1_DEV}" \
  --adapter-root "${PI_FLUX_ADAPTER_ROOT}" \
  --realesrgan-x2 "${REALESRGAN_X2}"
```

FLUX.2 Klein Base:

```bash
python examples/flux2_mrflow.py \
  --prompt "${PROMPT}" \
  --model "${FLUX2_KLEIN_BASE_9B}" \
  --realesrgan-x2 "${REALESRGAN_X2}" \
  --setting base9b_12plus1
```

FLUX.2 Klein non-base:

```bash
python examples/flux2_mrflow.py \
  --prompt "${PROMPT}" \
  --model "${FLUX2_KLEIN_9B}" \
  --realesrgan-x2 "${REALESRGAN_X2}" \
  --setting 9b_4plus1
```

Z-Image-Turbo:

```bash
python examples/zimage_turbo_mrflow.py \
  --prompt "${PROMPT}" \
  --model "${Z_IMAGE_TURBO}" \
  --realesrgan-x2 "${REALESRGAN_X2}" \
  --stage1-steps 9 \
  --refine-steps 9 \
  --strength 0.11
```

You can also edit all placeholder paths in `run_examples.sh` and run:

```bash
bash examples/run_examples.sh
```

## Outputs

Root-level demos write fixed filenames:

- `stage1_low.png`
- `stage2_upscaled.png`
- `stage3_refined.png`

Parameterized scripts in this directory add a descriptive prefix containing the model family, setting, seed, and resolution, for example:

- `qwen_image_mrflow_12plus1_seed2026_1024x1024_stage1_low.png`
- `qwen_image_mrflow_12plus1_seed2026_1024x1024_stage2_upscaled.png`
- `qwen_image_mrflow_12plus1_seed2026_1024x1024_stage3_refined.png`

The final image is always the `stage3_refined` file.
