from abc import ABC

import gym
import torch
import torch.nn as nn

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
        raise NotImplementedError
