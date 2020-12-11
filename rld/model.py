from abc import ABC
from collections import OrderedDict
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import tree
from gym import Space
from gym.spaces import Box, Dict
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor

from rld.exception import SpaceNotSupported
from rld.typing import (
    ObsLike,
    ObsLikeStrict,
    ObsTensorLike,
    ObsTensorStrict,
    HiddenState,
)


class Model(ABC, nn.Module):
    def input_device(self) -> torch.device:
        raise NotImplementedError

    def output_device(self) -> torch.device:
        return self.input_device()

    def action_space(self) -> Space:
        raise NotImplementedError

    def obs_space(self) -> Space:
        raise NotImplementedError

    # From PyTorch 1.6
    def _forward_unimplemented(self, *input: Any) -> None:
        pass


class RecurrentModel(Model, ABC):
    def initial_state(self) -> HiddenState:
        """
        [B x 2 x 1 x CELL_SIZE]
        """
        raise NotImplementedError

    def last_output_state(self) -> HiddenState:
        raise NotImplementedError

    def reshape_to_torch(self, state: HiddenState) -> HiddenState:
        return state.permute((1, 2, 0, 3))

    def reshape_to_store(self, state: HiddenState) -> HiddenState:
        return state.permute((2, 0, 1, 3))


RayModel = ModelV2


class RayModelWrapper(Model, ABC):
    def __init__(self, model: RayModel):
        super().__init__()
        self.model = model
        self.preprocessor = get_preprocessor(self.obs_space())(self.obs_space())

    def unwrapped(self):
        return self.model

    def input_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def action_space(self) -> Space:
        return self.model.action_space

    def obs_space(self) -> Space:
        if hasattr(self.model.obs_space, "original_space"):
            return self.model.obs_space.original_space
        else:
            return self.model.obs_space


class RayFeedforwardModelWrapper(RayModelWrapper):
    def forward(self, obs_flat: ObsTensorStrict):
        if isinstance(self.obs_space(), Box):
            # We need to unpack e.g. image-like observation,
            # as RLlib doesn't flatten them into 1D vectors
            input_dict = {
                "obs": unpack_tensor(obs_flat, self.obs_space()),
                "obs_flat": obs_flat,
            }
        else:
            input_dict = {"obs": obs_flat, "obs_flat": obs_flat}
        seq_lens = None
        state = None
        return self.model(input_dict, state, seq_lens)[0]


class RayRecurrentModelWrapper(RecurrentModel, RayModelWrapper):
    def __init__(self, model: RayModel, lstm_cell_size: int):
        super().__init__(model)
        self.lstm_cell_size = lstm_cell_size

        self._last_state: Optional[HiddenState] = None

    def forward(self, obs_flat: ObsTensorStrict, state: HiddenState):
        if isinstance(self.obs_space(), Box):
            # We need to unpack e.g. image-like observation,
            # as RLlib doesn't flatten them into 1D vectors
            input_dict = {
                "obs": unpack_tensor(obs_flat, self.obs_space()),
                "obs_flat": obs_flat,
            }
        else:
            input_dict = {"obs": obs_flat, "obs_flat": obs_flat}
        # For now we don't support batching multiple steps in a trajectory,
        # so the sequence length is always one, for each element in a batch
        batch_size = obs_flat.size(0)
        seq_lens = torch.ones(batch_size, dtype=torch.long, device=self.input_device())

        state = self.reshape_to_torch(state)
        logits, new_state = self.model(
            input_dict, [state[0].contiguous(), state[1].contiguous()], seq_lens
        )
        self._last_state = self.reshape_to_store(torch.stack(new_state))

        return logits

    def initial_state(self) -> HiddenState:
        initial_state = [
            torch.tensor(s, device=self.input_device())
            for s in self.model.get_initial_state()
        ]
        return torch.stack(initial_state).unsqueeze(dim=0).unsqueeze(dim=2)

    def last_output_state(self) -> HiddenState:
        if self._last_state is None:
            raise RuntimeError(
                "Trying to get last output hidden state without calling "
                "forward() first."
            )
        return self._last_state

    def reshape_to_torch(self, state: HiddenState) -> HiddenState:
        permuted = super().reshape_to_torch(state)
        return permuted.squeeze(dim=1)

    def reshape_to_store(self, state: HiddenState) -> HiddenState:
        return super().reshape_to_store(state.unsqueeze(dim=1))


def pack_array(obs: ObsLike, space: Space) -> ObsLikeStrict:
    if isinstance(space, Box):
        return np.asarray(obs, dtype=np.float32).flatten()
    elif isinstance(space, Dict):
        packed_values = [pack_array(obs[name], s) for name, s in space.spaces.items()]
        return np.concatenate(packed_values)
    else:
        raise SpaceNotSupported(space)


def unpack_array(obs: ObsLikeStrict, space: Space) -> ObsLike:
    if isinstance(space, Box):
        return np.asarray(obs).reshape(space.shape)
    elif isinstance(space, Dict):
        sizes = [_packed_size(s) for s in space.spaces.values()]
        split_packed = np.split(obs, np.cumsum(sizes)[:-1])
        split_unpacked = [
            (name, unpack_array(unpacked, s))
            for unpacked, (name, s) in zip(split_packed, space.spaces.items())
        ]
        return OrderedDict(split_unpacked)
    else:
        raise SpaceNotSupported(space)


def unpack_tensor(obs: ObsTensorStrict, space: Space) -> ObsTensorLike:
    batch_size = obs.size(0) if obs.ndim > 1 else None
    if batch_size is None:
        return _unpack_tensor_single(obs, space)
    else:
        return _unpack_tensor_batched(obs, space)


def _unpack_tensor_single(obs: ObsTensorStrict, space: Space) -> ObsTensorLike:
    if isinstance(space, Box):
        return obs.reshape(space.shape)
    elif isinstance(space, Dict):
        sizes = [_packed_size(s) for s in space.spaces.values()]
        split_packed = torch.split(obs, sizes)
        split_unpacked = [
            (name, unpack_tensor(unpacked, s))
            for unpacked, (name, s) in zip(split_packed, space.spaces.items())
        ]
        return OrderedDict(split_unpacked)
    else:
        raise SpaceNotSupported(space)


def _unpack_tensor_batched(obs: ObsTensorStrict, space: Space) -> ObsTensorLike:
    batch_size = obs.size(0)
    return _merge_unpacked_batch(
        [_unpack_tensor_single(obs[b], space) for b in range(batch_size)]
    )


def _merge_unpacked_batch(obs_list: List[ObsTensorLike]) -> ObsTensorLike:
    return tree.map_structure(lambda *elems: torch.stack(elems, dim=0), *obs_list)


def _packed_size(space: Space) -> int:
    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Dict):
        return int(sum([_packed_size(s) for s in space.spaces]))
    else:
        raise SpaceNotSupported(space)
