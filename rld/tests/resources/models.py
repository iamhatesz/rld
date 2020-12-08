from abc import ABC

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import flatten

from rld.model import Model
from rld.tests.resources.spaces import (
    DISCRETE_ACTION_SPACE,
    BOX_OBS_SPACE,
    MULTI_DISCRETE_ACTION_SPACE,
    IMAGE_OBS_SPACE,
    DICT_OBS_SPACE,
)


class ObsMixin:
    def init_hidden(self) -> nn.Module:
        raise NotImplementedError

    def preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ActionMixin:
    def init_head(self) -> nn.Module:
        raise NotImplementedError


class BaseModel(Model, ObsMixin, ActionMixin, ABC):
    NUM_HIDDEN_NEURONS = 64

    def __init__(self):
        super().__init__()

        self.hidden = self.init_hidden()
        self.head = self.init_head()

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        obs = self.preprocess_obs(obs_flat)
        x = F.relu(self.hidden(obs))
        x = self.head(x)
        return x

    def input_device(self) -> torch.device:
        return torch.device("cpu")


class BaseRecurrentModel(Model, ObsMixin, ActionMixin, ABC):
    NUM_HIDDEN_NEURONS = 64

    def __init__(self):
        super().__init__()

        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(self.NUM_HIDDEN_NEURONS, self.NUM_HIDDEN_NEURONS)
        self.head = self.init_head()

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        obs = self.preprocess_obs(obs_flat)
        x = F.relu(self.hidden(obs))

        x = self.head(x)
        return x

    def input_device(self) -> torch.device:
        return torch.device("cpu")


class BoxObsMixin(ObsMixin):
    def init_hidden(self: BaseModel) -> nn.Module:
        return nn.Linear(self.obs_space().sample().size, self.NUM_HIDDEN_NEURONS)

    def preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs


class ImageObsMixin(ObsMixin):
    def init_hidden(self: BaseModel) -> nn.Module:
        image_size = self.obs_space().shape[:2]
        return nn.Sequential(
            nn.Conv2d(4, self.NUM_HIDDEN_NEURONS, kernel_size=image_size), nn.Flatten(),
        )

    def preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.permute((0, 3, 1, 2))


class DictObxMixin(ObsMixin):
    def init_hidden(self: BaseModel) -> nn.Module:
        num_inputs = flatten(self.obs_space(), self.obs_space().sample()).size
        return nn.Linear(num_inputs, self.NUM_HIDDEN_NEURONS)

    def preprocess_obs(self: BaseModel, obs: torch.Tensor) -> torch.Tensor:
        return obs


class DiscreteActionMixin(ActionMixin):
    def init_head(self: BaseModel) -> nn.Module:
        return nn.Linear(self.NUM_HIDDEN_NEURONS, self.action_space().n)


class MultiDiscreteActionMixin(ActionMixin):
    def init_head(self: BaseModel) -> nn.Module:
        return nn.Linear(self.NUM_HIDDEN_NEURONS, sum(self.action_space().nvec))


class BoxObsDiscreteActionModel(BaseModel, BoxObsMixin, DiscreteActionMixin):
    def obs_space(self) -> gym.Space:
        return BOX_OBS_SPACE

    def action_space(self) -> gym.Space:
        return DISCRETE_ACTION_SPACE


class BoxObsMultiDiscreteActionModel(BaseModel, BoxObsMixin, MultiDiscreteActionMixin):
    def obs_space(self) -> gym.Space:
        return BOX_OBS_SPACE

    def action_space(self) -> gym.Space:
        return MULTI_DISCRETE_ACTION_SPACE


class ImageObsDiscreteActionModel(BaseModel, ImageObsMixin, DiscreteActionMixin):
    def obs_space(self) -> gym.Space:
        return IMAGE_OBS_SPACE

    def action_space(self) -> gym.Space:
        return DISCRETE_ACTION_SPACE


class ImageObsMultiDiscreteActionModel(
    BaseModel, ImageObsMixin, MultiDiscreteActionMixin
):
    def obs_space(self) -> gym.Space:
        return IMAGE_OBS_SPACE

    def action_space(self) -> gym.Space:
        return MULTI_DISCRETE_ACTION_SPACE


class DictObsDiscreteActionModel(BaseModel, DictObxMixin, DiscreteActionMixin):
    def obs_space(self) -> gym.Space:
        return DICT_OBS_SPACE

    def action_space(self) -> gym.Space:
        return DISCRETE_ACTION_SPACE


class DictObsMultiDiscreteActionModel(
    BaseModel, DictObxMixin, MultiDiscreteActionMixin
):
    def obs_space(self) -> gym.Space:
        return DICT_OBS_SPACE

    def action_space(self) -> gym.Space:
        return MULTI_DISCRETE_ACTION_SPACE


ALL_MODELS = [
    BoxObsDiscreteActionModel,
    BoxObsMultiDiscreteActionModel,
    ImageObsDiscreteActionModel,
    ImageObsMultiDiscreteActionModel,
    DictObsDiscreteActionModel,
    DictObsMultiDiscreteActionModel,
]
