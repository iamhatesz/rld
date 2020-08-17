from abc import ABC
from typing import Any

import gym
import torch
import torch.nn as nn
from gym.spaces import flatten, unflatten

from rld.typing import ObsLike, ObsLikeStrict


class Model(ABC, nn.Module):
    def input_device(self) -> torch.device:
        raise NotImplementedError

    def output_device(self) -> torch.device:
        return self.input_device()

    def action_space(self) -> gym.Space:
        raise NotImplementedError

    def obs_space(self) -> gym.Space:
        raise NotImplementedError

    def flatten_obs(self, obs: ObsLike) -> ObsLikeStrict:
        if isinstance(self.obs_space(), gym.spaces.Box):
            return obs
        return flatten(self.obs_space(), obs)

    def unflatten_obs(self, obs: ObsLikeStrict) -> ObsLike:
        if isinstance(self.obs_space(), gym.spaces.Box):
            return obs
        return unflatten(self.obs_space(), obs)

    # From PyTorch 1.6
    def _forward_unimplemented(self, *input: Any) -> None:
        pass
