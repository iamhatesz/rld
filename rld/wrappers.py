import gym
import torch
from gym.spaces import flatten, unflatten
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor

from rld.model import Model
from rld.typing import ObsLikeStrict, ObsLike


RayModel = ModelV2


class RayModelWrapper(Model):
    def __init__(self, model: RayModel):
        super().__init__()
        self.model = model
        self.preprocessor = get_preprocessor(self.original_obs_space())(
            self.original_obs_space()
        )

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
        return self.model.obs_space

    def original_obs_space(self) -> gym.Space:
        if hasattr(self.obs_space(), "original_space"):
            return self.obs_space().original_space
        else:
            return self.obs_space()

    def flatten_obs(self, obs: ObsLike) -> ObsLikeStrict:
        if isinstance(self.original_obs_space(), gym.spaces.Box):
            return obs
        return flatten(self.original_obs_space(), obs)

    def unflatten_obs(self, obs: ObsLikeStrict) -> ObsLike:
        if isinstance(self.original_obs_space(), gym.spaces.Box):
            return obs
        return unflatten(self.original_obs_space(), obs)
