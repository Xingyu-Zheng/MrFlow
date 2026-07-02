import sys
import types
import os
from pathlib import Path

import torch


LAKONLAB_ROOT = Path(
    os.environ.get("LAKONLAB_ROOT", Path(__file__).resolve().parent / "third_party" / "LakonLab")
)


def add_lakonlab_to_path():
    if not LAKONLAB_ROOT.exists():
        raise FileNotFoundError(
            "LakonLab is required for PiFlow examples. Set LAKONLAB_ROOT to the LakonLab repository path."
        )
    root = str(LAKONLAB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def install_lakonlab_inference_shims():
    """Avoid importing LakonLab training-only modules that need optional deps."""
    add_lakonlab_to_path()
    models_path = LAKONLAB_ROOT / "lakonlab" / "models"
    diffusions_path = models_path / "diffusions"
    architectures_path = models_path / "architectures"
    runner_path = LAKONLAB_ROOT / "lakonlab" / "runner"

    if "lakonlab.models" not in sys.modules:
        module = types.ModuleType("lakonlab.models")
        module.__path__ = [str(models_path)]
        sys.modules["lakonlab.models"] = module
    if "lakonlab.models.diffusions" not in sys.modules:
        module = types.ModuleType("lakonlab.models.diffusions")
        module.__path__ = [str(diffusions_path)]
        sys.modules["lakonlab.models.diffusions"] = module
    if "lakonlab.models.architectures" not in sys.modules:
        module = types.ModuleType("lakonlab.models.architectures")
        module.__path__ = [str(architectures_path)]
        sys.modules["lakonlab.models.architectures"] = module
    for package_name, package_path in {
        "lakonlab.models.architectures.gmflow": architectures_path / "gmflow",
        "lakonlab.models.architectures.dxflow": architectures_path / "dxflow",
        "lakonlab.models.architectures.asymflow": architectures_path / "asymflow",
        "lakonlab.models.architectures.diffusers": architectures_path / "diffusers",
    }.items():
        if package_name not in sys.modules:
            module = types.ModuleType(package_name)
            module.__path__ = [str(package_path)]
            sys.modules[package_name] = module
    if "lakonlab.runner" not in sys.modules:
        module = types.ModuleType("lakonlab.runner")
        module.__path__ = [str(runner_path)]
        sys.modules["lakonlab.runner"] = module
    if "lakonlab.utils" not in sys.modules:
        module = types.ModuleType("lakonlab.utils")
        module.__path__ = [str(LAKONLAB_ROOT / "lakonlab" / "utils")]
        module.get_root_logger = _get_root_logger
        module.rgetattr = _rgetattr
        sys.modules["lakonlab.utils"] = module
    if "lakonlab.runner.checkpoint" not in sys.modules:
        module = types.ModuleType("lakonlab.runner.checkpoint")
        module._load_cached_checkpoint = _load_cached_checkpoint
        module.load_full_state_dict = _load_full_state_dict
        module.load_checkpoint = _load_cached_checkpoint
        sys.modules["lakonlab.runner.checkpoint"] = module
    _install_diffusion_gmflow_shim()


class _NoopRegistry:
    def register_module(self, *args, **kwargs):
        def wrap(cls):
            return cls

        if args and isinstance(args[0], type):
            return args[0]
        return wrap


def _get_root_logger(*args, **kwargs):
    import logging

    return logging.getLogger("lakonlab")


def _rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    for name in attr.split("."):
        obj = _getattr(obj, name)
    return obj


def _load_cached_checkpoint(filename, map_location=None, logger=None):
    from safetensors.torch import load_file

    filename = str(filename)
    if filename.endswith(".safetensors"):
        return load_file(filename, device=map_location or "cpu")
    return torch.load(filename, map_location=map_location or "cpu")


def _load_full_state_dict(module, state_dict, strict=False, logger=None, assign=False):
    try:
        missing, unexpected = module.load_state_dict(state_dict, strict=strict, assign=assign)
    except TypeError:
        missing, unexpected = module.load_state_dict(state_dict, strict=strict)
    if logger is not None and (missing or unexpected):
        logger.warning("Loaded with missing=%s unexpected=%s", missing, unexpected)


def _install_builder_shim():
    if "lakonlab.models.builder" in sys.modules:
        return
    module = types.ModuleType("lakonlab.models.builder")
    module.MODULES = _NoopRegistry()
    module.MODELS = module.MODULES
    module.build_module = lambda cfg, default_args=None: None
    module.build_model = module.build_module
    sys.modules["lakonlab.models.builder"] = module


def _install_diffusion_gmflow_shim():
    if "lakonlab.models.diffusions.gmflow" in sys.modules:
        return

    def gmflow_posterior_mean_jit(
        sigma_t_src,
        sigma_t,
        x_t_src,
        x_t,
        gm_means,
        gm_vars,
        gm_logweights,
        eps: float,
        gm_dim: int = -4,
        channel_dim: int = -3,
    ):
        sigma_t_src = sigma_t_src.clamp(min=eps)
        sigma_t = sigma_t.clamp(min=eps)
        alpha_t_src = 1 - sigma_t_src
        alpha_t = 1 - sigma_t
        alpha_over_sigma_t_src = alpha_t_src / sigma_t_src
        alpha_over_sigma_t = alpha_t / sigma_t
        zeta = alpha_over_sigma_t.square() - alpha_over_sigma_t_src.square()
        nu = alpha_over_sigma_t * x_t / sigma_t - alpha_over_sigma_t_src * x_t_src / sigma_t_src
        nu = nu.unsqueeze(gm_dim)
        zeta = zeta.unsqueeze(gm_dim)
        denom = (gm_vars * zeta + 1).clamp(min=eps)
        out_means = (gm_vars * nu + gm_means) / denom
        logweights_delta = (gm_means * (nu - 0.5 * zeta * gm_means)).sum(
            dim=channel_dim, keepdim=True
        ) / denom
        out_weights = (gm_logweights + logweights_delta).softmax(dim=gm_dim)
        return (out_means * out_weights).sum(dim=gm_dim)

    module = types.ModuleType("lakonlab.models.diffusions.gmflow")
    module.gmflow_posterior_mean_jit = gmflow_posterior_mean_jit
    sys.modules["lakonlab.models.diffusions.gmflow"] = module


def import_piflow_components():
    install_lakonlab_inference_shims()
    _install_builder_shim()
    _install_optional_architecture_shims()
    from lakonlab.models.diffusions.schedulers.flow_map_sde import FlowMapSDEScheduler
    from lakonlab.pipelines.pipeline_piflux import PiFluxPipeline
    from lakonlab.pipelines.pipeline_piqwen import PiQwenImagePipeline

    return FlowMapSDEScheduler, PiFluxPipeline, PiQwenImagePipeline


def _install_optional_architecture_shims():
    placeholders = {
        "lakonlab.models.architectures.gmflow.gmflux2": "_GMFlux2Transformer2DModel",
        "lakonlab.models.architectures.asymflow.asymflux2": "_AsymFlux2Transformer2DModel",
    }
    for module_name, class_name in placeholders.items():
        if module_name in sys.modules:
            continue
        module = types.ModuleType(module_name)

        class _UnusedArchitecture:
            pass

        setattr(module, class_name, _UnusedArchitecture)
        sys.modules[module_name] = module


def adapter_policy(adapter_name):
    if adapter_name.startswith("gm"):
        return "GMFlow", None
    if adapter_name.startswith("dxqwen_p"):
        return "DX", {"mode": "polynomial", "shift": 3.2, "segment_size": 0.1}
    if adapter_name.startswith("dx"):
        return "DX", {"mode": "grid", "shift": 3.2, "segment_size": 0.1}
    raise ValueError(f"Cannot infer PiFlow policy from adapter name: {adapter_name}")


def make_flow_map_scheduler(pipe, shift=3.2, final_step_size_scale=0.5):
    FlowMapSDEScheduler, _, _ = import_piflow_components()
    scheduler = FlowMapSDEScheduler.from_config(
        pipe.scheduler.config,
        shift=shift,
        use_dynamic_shifting=False,
        final_step_size_scale=final_step_size_scale,
    )
    scheduler.num_timesteps = scheduler.config.num_train_timesteps
    return scheduler


def clean_prompt(prompt):
    return " ".join(line.strip().strip('"') for line in prompt.strip().splitlines() if line.strip())
