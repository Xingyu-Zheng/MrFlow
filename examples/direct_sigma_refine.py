from contextlib import contextmanager
from types import MethodType

import torch


def direct_sigma_nodes(first_sigma: float, steps: int) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    return torch.linspace(float(first_sigma), 0.0, steps + 1, dtype=torch.float32)


@contextmanager
def force_direct_sigma_set_timesteps(scheduler, first_sigma: float, steps: int, device):
    original = scheduler.set_timesteps
    force_device = device

    def set_timesteps(self, sigmas=None, device=None, **kwargs):
        _ = sigmas
        sigmas = direct_sigma_nodes(first_sigma, steps).to(device or force_device)
        self.num_inference_steps = steps
        self.timesteps = sigmas[:-1] * self.config.num_train_timesteps
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None
        if hasattr(self, "set_begin_index"):
            self.set_begin_index(0)

    scheduler.set_timesteps = MethodType(set_timesteps, scheduler)
    try:
        yield
    finally:
        scheduler.set_timesteps = original
