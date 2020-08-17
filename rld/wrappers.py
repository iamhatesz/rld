import gym
import torch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor

from rld.model import Model

RayModel = ModelV2


class RayModelWrapper(Model):
    def __init__(self, model: RayModel):
        super().__init__()
        self.model = model
        self.preprocessor = get_preprocessor(self.obs_space())(self.obs_space())

    def unwrapped(self):
        return self.model

    def forward(self, x):
        input_dict = {"obs": x, "obs_flat": x}
        state = None
        seq_lens = None
        return self.model(input_dict, state, seq_lens)[0]

    def input_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def action_space(self) -> gym.Space:
        return self.model.action_space

    def obs_space(self) -> gym.Space:
        if hasattr(self.model.obs_space, "original_space"):
            return self.model.obs_space.original_space
        else:
            return self.model.obs_space
