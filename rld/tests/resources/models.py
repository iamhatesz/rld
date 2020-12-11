from abc import ABC
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space

from rld.model import Model, RecurrentModel, unpack_tensor
from rld.tests.resources.spaces import (
    DISCRETE_ACTION_SPACE,
    BOX_OBS_SPACE,
    MULTI_DISCRETE_ACTION_SPACE,
    IMAGE_OBS_SPACE,
    DICT_OBS_SPACE,
)
from rld.typing import HiddenState, ObsTensorStrict, ObsTensorLike


class ObsMixin:
    def init_hidden(self) -> nn.Module:
        raise NotImplementedError

    def call_hidden(self, obs: ObsTensorLike) -> torch.Tensor:
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

    def forward(self, obs_flat: ObsTensorStrict) -> torch.Tensor:
        obs = unpack_tensor(obs_flat, self.obs_space())
        obs = self.preprocess_obs(obs)
        x = F.relu(self.call_hidden(obs))
        x = self.head(x)
        return x

    def input_device(self) -> torch.device:
        return torch.device("cpu")


class BaseRecurrentModel(RecurrentModel, ObsMixin, ActionMixin, ABC):
    NUM_HIDDEN_NEURONS = 64

    def __init__(self):
        super().__init__()

        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(
            self.NUM_HIDDEN_NEURONS, self.NUM_HIDDEN_NEURONS, batch_first=True
        )
        self.head = self.init_head()

        self._last_state: Optional[HiddenState] = None

    def forward(self, obs_flat: ObsTensorStrict, state: HiddenState) -> torch.Tensor:
        obs = unpack_tensor(obs_flat, self.obs_space())
        obs = self.preprocess_obs(obs)

        x = F.relu(self.call_hidden(obs))
        x = x.unsqueeze(dim=1)

        state = self.reshape_to_torch(state)
        x, state = self.lstm(x, (state[0], state[1]))
        state = self.reshape_to_store(torch.stack(state))

        x = x.squeeze(dim=1)
        x = self.head(x)

        self._last_state = state
        return x

    def input_device(self) -> torch.device:
        return torch.device("cpu")

    def initial_state(self) -> HiddenState:
        return torch.zeros(
            (1, 2, 1, self.NUM_HIDDEN_NEURONS),
            dtype=torch.float32,
            device=self.input_device(),
        )

    def last_output_state(self) -> HiddenState:
        if self._last_state is None:
            raise RuntimeError(
                "Trying to get last output hidden state without calling "
                "forward() first."
            )
        return self._last_state


class BoxObsMixin(ObsMixin):
    def obs_space(self) -> Space:
        return BOX_OBS_SPACE

    def init_hidden(self: BaseModel) -> nn.Module:
        return nn.Linear(self.obs_space().sample().size, self.NUM_HIDDEN_NEURONS)

    def call_hidden(self: BaseModel, obs: ObsTensorLike) -> torch.Tensor:
        return self.hidden(obs)

    def preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs


class ImageObsMixin(ObsMixin):
    def obs_space(self) -> Space:
        return IMAGE_OBS_SPACE

    def init_hidden(self: BaseModel) -> nn.Module:
        image_size = self.obs_space().shape[:2]
        return nn.Sequential(
            nn.Conv2d(4, self.NUM_HIDDEN_NEURONS, kernel_size=image_size), nn.Flatten(),
        )

    def call_hidden(self: BaseModel, obs: ObsTensorLike) -> torch.Tensor:
        return self.hidden(obs)

    def preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.permute((0, 3, 1, 2))


class DictObxMixin(ObsMixin):
    def obs_space(self) -> Space:
        return DICT_OBS_SPACE

    def init_hidden(self: BaseModel) -> nn.Module:
        return nn.ModuleDict(
            {
                "a": nn.Sequential(
                    nn.Flatten(), nn.Linear(24, self.NUM_HIDDEN_NEURONS),
                ),
                "b": nn.Linear(2, self.NUM_HIDDEN_NEURONS),
            }
        )

    def call_hidden(self: BaseModel, obs: ObsTensorLike) -> torch.Tensor:
        return self.hidden["a"](obs["a"]) + self.hidden["b"](obs["b"])

    def preprocess_obs(self: BaseModel, obs: torch.Tensor) -> torch.Tensor:
        return obs


class DiscreteActionMixin(ActionMixin):
    def action_space(self) -> Space:
        return DISCRETE_ACTION_SPACE

    def init_head(self: BaseModel) -> nn.Module:
        return nn.Linear(self.NUM_HIDDEN_NEURONS, self.action_space().n)


class MultiDiscreteActionMixin(ActionMixin):
    def action_space(self) -> Space:
        return MULTI_DISCRETE_ACTION_SPACE

    def init_head(self: BaseModel) -> nn.Module:
        return nn.Linear(self.NUM_HIDDEN_NEURONS, sum(self.action_space().nvec))


class BoxObsDiscreteActionModel(BoxObsMixin, DiscreteActionMixin, BaseModel):
    pass


class BoxObsMultiDiscreteActionModel(BoxObsMixin, MultiDiscreteActionMixin, BaseModel):
    pass


class ImageObsDiscreteActionModel(ImageObsMixin, DiscreteActionMixin, BaseModel):
    pass


class ImageObsMultiDiscreteActionModel(
    ImageObsMixin, MultiDiscreteActionMixin, BaseModel
):
    pass


class DictObsDiscreteActionModel(DictObxMixin, DiscreteActionMixin, BaseModel):
    pass


class DictObsMultiDiscreteActionModel(
    DictObxMixin, MultiDiscreteActionMixin, BaseModel
):
    pass


class BoxObsDiscreteActionRecurrentModel(
    BoxObsMixin, DiscreteActionMixin, BaseRecurrentModel
):
    pass


class BoxObsMultiDiscreteActionRecurrentModel(
    BoxObsMixin, MultiDiscreteActionMixin, BaseRecurrentModel
):
    pass


class ImageObsDiscreteActionRecurrentModel(
    ImageObsMixin, DiscreteActionMixin, BaseRecurrentModel
):
    pass


class ImageObsMultiDiscreteActionRecurrentModel(
    ImageObsMixin, MultiDiscreteActionMixin, BaseRecurrentModel
):
    pass


class DictObsDiscreteActionRecurrentModel(
    DictObxMixin, DiscreteActionMixin, BaseRecurrentModel
):
    pass


class DictObsMultiDiscreteActionRecurrentModel(
    DictObxMixin, MultiDiscreteActionMixin, BaseRecurrentModel
):
    pass


ALL_MODELS = [
    BoxObsDiscreteActionModel,
    BoxObsMultiDiscreteActionModel,
    ImageObsDiscreteActionModel,
    ImageObsMultiDiscreteActionModel,
    DictObsDiscreteActionModel,
    DictObsMultiDiscreteActionModel,
    # Recurrent models
    BoxObsDiscreteActionRecurrentModel,
    BoxObsMultiDiscreteActionRecurrentModel,
    ImageObsDiscreteActionRecurrentModel,
    ImageObsMultiDiscreteActionRecurrentModel,
    DictObsDiscreteActionRecurrentModel,
    DictObsMultiDiscreteActionRecurrentModel,
]
