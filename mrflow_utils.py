from contextlib import contextmanager
from types import MethodType

import torch


@contextmanager
def direct_sigma_schedule(scheduler, first_sigma, steps, device):
    original = scheduler.set_timesteps
    force_device = device

    def set_timesteps(self, sigmas=None, device=None, **kwargs):
        _ = sigmas
        target_device = device or force_device
        sigmas = torch.linspace(first_sigma, 0.0, steps + 1, device=target_device)
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
