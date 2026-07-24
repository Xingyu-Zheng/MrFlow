# Multi-Resolution Flow Matching: Training-Free Diffusion Acceleration via Staged Sampling

<div align="center">

### Official implementation for MrFlow

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Trending%20Papers-ffd21e.svg?logo=huggingface)](https://huggingface.co/papers/trending)
[![Papers with Code: #3 on OneIG-EN](https://paperswithcode.co/api/v1/papers/2607.01642/leaderboard-badge.svg?eval=16706&live=1)](https://paperswithcode.co/benchmark/oneig-en?task=image-generation&eval=16706)
<br> 
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2607.01642-b31b1b.svg)](https://arxiv.org/abs/2607.01642)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Daily%20Papers-ffd21e.svg?logo=huggingface)](https://huggingface.co/papers/2607.01642)
[![Method](https://img.shields.io/badge/method-training--free-4c6fff.svg)](#highlights)
[![Acceleration](https://img.shields.io/badge/speedup-10x%2B-green.svg)](#results)
[![Backbones](https://img.shields.io/badge/backbones-FLUX%20%7C%20Qwen--Image%20%7C%20Z--Image-555.svg)](#supported-demos)

</div>

This repository provides the implementation of **MrFlow**, a training-free staged sampling method for accelerating pretrained flow-matching text-to-image diffusion models.

Clone the repository with submodules to also fetch the exact Real-ESRGAN source used by the paper experiments:

```bash
git clone --recursive https://github.com/Xingyu-Zheng/MrFlow.git
```

If you already cloned the repository, initialize the submodules with `git submodule update --init --recursive`.

MrFlow first samples a low-resolution image, upsamples the decoded result in pixel space with Real-ESRGAN, re-encodes the upsampled image, injects scheduler-consistent low-strength noise, and performs a short high-resolution refinement. The pipeline shifts most denoising cost from expensive high-resolution sampling to cheaper low-resolution sampling while preserving local detail quality.

<p align="center">
  <img src="assets/framework.webp" width="96%" alt="MrFlow framework">
</p>

## ✨ Highlights

- **Training-free deployment.** No finetuning, learned upsampler, or model-specific retraining is required.
- **No custom kernels.** The implementation uses standard PyTorch, Diffusers pipelines, and scheduler controls.
- **Strong aggressive-speed regime.** MrFlow reaches more than `10x` end-to-end speedup on Qwen-Image while preserving visual quality.
- **Works with distilled models.** The same pipeline can be combined with pretrained timestep-distilled models such as Pi-Flow and FLUX-schnell.
- **Compact staged design.** The implementation transfers across Qwen-Image, FLUX.1-dev, FLUX.2 Klein, and Z-Image families.

## 📢 News

- [2026/07] 🏆 MrFlow ranks in the Top 3 on the [OneIG-EN leaderboard](https://paperswithcode.co/benchmark/oneig-en?task=image-generation&eval=16706) on Hugging Face's Papers with Code.
- [2026/07] 💬 We open [Discussions](https://github.com/Xingyu-Zheng/MrFlow/discussions) for broader method discussions, prior-work comparisons, workflow/runtime observations, and community experiences. Feel free to discuss them there.
- [2026/07] 📰 MrFlow is listed on [Hugging Face Trending Papers](https://huggingface.co/papers/trending).
- [2026/07] 💡 We add a [Practical Tips](#practical-tips) section and encourage everyone to share useful observations and takeaways with each other.
- [2026/07] 🌱 We add a [community contribution area](community/) and welcome developers to share MrFlow ports, workflows, and experiments with each other.
- [2026/07] 📰 MrFlow is featured on [Hugging Face Daily Papers](https://huggingface.co/papers/2607.01642).
- [2026/07] ⚡ We release the MrFlow ComfyUI plugin.
- [2026/07] 🔥 The MrFlow paper is available on [arXiv](https://arxiv.org/abs/2607.01642), and the source code is released.

## 🛠️ Installation

Create a Diffusers-compatible environment for the target backbone. The demos use:

- PyTorch
- Diffusers
- Transformers
- Real-ESRGAN

MrFlow uses Real-ESRGAN for x2 pixel-space super-resolution. The repository includes the Real-ESRGAN source as a Git submodule at [`Real-ESRGAN/`](Real-ESRGAN/) for the paper experiments. The corresponding model weights are available separately:

```text
Code:    https://github.com/ai-forever/Real-ESRGAN
Weights: https://huggingface.co/ai-forever/Real-ESRGAN
```

The scripts contain placeholder checkpoint paths. Replace them with local paths to the pretrained text-to-image model and Real-ESRGAN x2 weights before running.

## 🚀 Quick Start

The repository root keeps only two minimal reference scripts plus the shared scheduler helper:

| Script | Model | Setting | Output |
| --- | --- | --- | --- |
| `qwen_image_mrflow.py` | Qwen-Image | MrFlow `12plus1` | `outputs/qwen_image_mrflow_12plus1/` |
| `flux1_mrflow.py` | FLUX.1-dev | MrFlow `12plus1` | `outputs/flux1_mrflow_12plus1/` |

Edit the checkpoint paths at the top of each script:

```python
MODEL = "/path/to/Qwen-Image"
REALESRGAN_X2 = "/path/to/RealESRGAN_x2.pth"
```

Run:

```bash
python qwen_image_mrflow.py
```

```bash
python flux1_mrflow.py
```

Each script saves:

- `stage1_low.png`: low-resolution generated image.
- `stage2_upscaled.png`: Real-ESRGAN x2 upsampled image.
- `stage3_refined.png`: final high-resolution refined image.

## ⚙️ Core Settings

| Setting | Low-resolution steps | Refinement steps | Direct sigma | Typical use |
| --- | ---: | ---: | ---: | --- |
| `12plus1` | 12 | 1 | `0.12` | Aggressive acceleration. |
| `20plus1` | 20 | 1 | `0.12` | Higher-quality operating point. |

The high-resolution refinement uses an explicit direct-sigma schedule. For example, `12plus1` denotes 12 low-resolution denoising steps followed by one high-resolution step from `sigma=0.12` to `0`.

Representative end-to-end speedups:

| Backbone | Setting | End-to-end speedup |
| --- | ---: | ---: |
| FLUX.1-dev | `12 + 1` | `8.25x` |
| Qwen-Image | `12 + 1` | `10.3x` |
| FLUX.2 Klein Base 9B | `12 + 1` | `8.79x` |
| Z-Image-Turbo | `8 + 1` | `21.0x` |
| Qwen-Image + Pi-Flow | `4 + 1` | up to `25x` |

Speedups are measured end to end, including text encoding, VAE encode/decode, super-resolution, noise preparation, and diffusion forward passes.

## 📦 Supported Demos

Parameterized variants and additional model-family demos are available in `examples/`.

| Script | Backbone | Notes |
| --- | --- | --- |
| `examples/flux1_mrflow.py` | FLUX.1-dev | Training-free MrFlow. |
| `examples/flux1_piflow_mrflow.py` | FLUX.1-dev + Pi-Flow | Combines MrFlow with distilled weights. |
| `examples/qwen_image_mrflow.py` | Qwen-Image | Training-free MrFlow. |
| `examples/qwen_image_piflow_mrflow.py` | Qwen-Image + Pi-Flow | Combines MrFlow with distilled weights. |
| `examples/flux2_mrflow.py` | FLUX.2 Klein | Base and non-base variants. |
| `examples/zimage_turbo_mrflow.py` | Z-Image-Turbo | Reduced-step model plus MrFlow refinement. |

Run all configured examples with:

```bash
bash examples/run_examples.sh
```

See [examples/README.md](examples/README.md) for command-line usage, FLUX.2 Klein presets, Z-Image-Turbo refinement defaults, and output filename conventions.

Pi-Flow examples are optional and require a separate local checkout of [LakonLab](https://github.com/Lakonik/LakonLab). Set `LAKONLAB_ROOT` to that checkout before running the Pi-Flow scripts.

## 🖼️ Results

**Qwen-Image generation examples.** With 12 low-resolution steps and one high-resolution refinement step, MrFlow produces diverse 1024-resolution samples on Qwen-Image while reaching above `10x` end-to-end speedup.

<p align="center">
  <img src="assets/showcase-mrflow-qwen.jpg" width="96%" alt="MrFlow Qwen-Image samples">
</p>

**Accuracy-efficiency trade-off.** On FLUX.1-dev and Qwen-Image, MrFlow offers a flexible trade-off between generation quality and measured end-to-end speedup, and remains effective where other training-free strategies degrade sharply.

<p align="center">
  <img src="assets/tradeoff.webp" width="96%" alt="MrFlow trade-off curve">
</p>

**Runtime breakdown.** For Qwen-Image `12plus1`, measured end-to-end latency is `4.77s` versus `49.32s` for native 50-step inference. The main cost is shifted from high-resolution sampling to cheaper low-resolution sampling, while SR and VAE overhead remain small.

<p align="center">
  <img src="assets/efficiency.webp" width="96%" alt="MrFlow runtime breakdown">
</p>

## 🧩 ComfyUI Plugin

<p align="center">
  <img src="assets/comfyui.webp" width="96%" alt="MrFlow ComfyUI Plugin">
</p>

The repository also includes `ComfyUI-MrFlow/`, a ComfyUI custom-node extension for Qwen-oriented MrFlow workflows. It provides helper nodes, editable workflow and API JSON examples, a reusable subgraph, and a model-link helper for split Qwen-Image bundles.

To use it, place or symlink `ComfyUI-MrFlow/` into `ComfyUI/custom_nodes/`, restart ComfyUI, and open `ComfyUI-MrFlow/examples/qwen_mrflow_workflow.json` or load `ComfyUI-MrFlow/subgraphs/qwen_mrflow.json`.

## 🌱 Community Contributions

We have seen strong community interest in adapting MrFlow to additional model families, ComfyUI loaders, and local workflows. To make these efforts easier to share early, community contributions are collected in [`community/experimental/`](community/experimental/) before selected pieces are tested and polished. Contributions that are ready for broader reuse may later move to a sibling `community/verified/` area, or be promoted into the main examples or plugin folders when they become part of the official workflow.

Pull requests are preferred because they are easier to review and track. If you are not familiar with GitHub PRs, it is also fine to open an issue, link your code or workflow, and tag the maintainers directly.

<a id="practical-tips"></a>
## 💡 Practical Tips

A few notes from our open-source release and community testing:

1. After releasing the open-source version, we have found that keeping the high-resolution refinement to a single step while using a larger direct sigma, such as `0.16`-`0.20`, can often improve visual quality, especially for generations that include text. If this matches your experience, feel free to share your feedback anytime.
2. [RealRebelAI](https://github.com/RealRebelAI) appears to have found strong MrFlow results on Krea-2, one of the newer state-of-the-art models, and we encourage everyone to try MrFlow on more recent models and share what they discover.
3. [RealRebelAI](https://github.com/RealRebelAI) also found that using `4x Foolhardy Remacri` for 2048-resolution output can work well with MrFlow. We encourage everyone to try different super-resolution ratios or stronger super-resolution models, since this kind of exploration is very much in the spirit of MrFlow.

## 📝 Citation

If you find MrFlow useful, please cite our paper:

```bibtex
@misc{zheng2026multiresolutionflowmatchingtrainingfree,
  title={Multi-Resolution Flow Matching: Training-Free Diffusion Acceleration via Staged Sampling},
  author={Xingyu Zheng and Xianglong Liu and Yifu Ding and Weilun Feng and Junqing Lin and Jinyang Guo and Haotong Qin},
  year={2026},
  eprint={2607.01642},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2607.01642},
}
```

## 🙏 Acknowledgements

This implementation builds on the Diffusers ecosystem and uses [ai-forever/Real-ESRGAN](https://github.com/ai-forever/Real-ESRGAN) for pixel-space super-resolution.
