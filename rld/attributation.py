import sys
from collections import abc
from dataclasses import dataclass, replace
from enum import IntEnum
from functools import partial
from typing import Optional, Union

import gym
import numpy as np
import torch
from captum.attr import IntegratedGradients
from gym.spaces import flatten, unflatten
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.utils.torch_ops import convert_to_non_torch_type

from rld.exception import ActionSpaceNotSupported, EnumValueNotFound
from rld.model import Model
from rld.rollout import (
    Trajectory,
    Timestep,
    DiscreteActionAttributation,
    MultiDiscreteActionAttributation,
    TupleActionAttributation,
    ActionAttributation,
    remove_channel_dim_from_image_space,
)
from rld.typing import BaselineBuilder, ObsLike, AttributationLike


class AttributationVisualizationSign(IntEnum):
    ALL = 0
    POSITIVE = 1
    NEGATIVE = 2
    ABSOLUTE_VALUE = 3


class AttributationProcessor:
    def transform(self, attr: ActionAttributation) -> ActionAttributation:
        return attr.map(self._transform)

    def _transform(self, attr: AttributationLike) -> AttributationLike:
        raise NotImplementedError


class NormalizeAttributationProcessor(AttributationProcessor):
    """
    Based on Captum image attributation visualization technique.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        obs_is_image: bool = False,
        sign: AttributationVisualizationSign = AttributationVisualizationSign.ALL,
        outlier_percentile: Union[int, float] = 5,
    ):
        self.obs_space = obs_space
        self.obs_is_image = obs_is_image
        self.sign = sign
        self.outlier_percentile = outlier_percentile

    def _transform(self, attr: AttributationLike) -> AttributationLike:
        obs_space = self.obs_space
        if self.obs_is_image:
            attr = np.sum(attr, axis=2)
            obs_space = remove_channel_dim_from_image_space(obs_space)
        attr = flatten(self.obs_space, attr)
        if self.sign == AttributationVisualizationSign.ALL:
            scaling_factor = self._calculate_safe_scaling_factor(np.abs(attr))
        elif self.sign == AttributationVisualizationSign.POSITIVE:
            attr = (attr > 0) * attr
            scaling_factor = self._calculate_safe_scaling_factor(attr)
        elif self.sign == AttributationVisualizationSign.NEGATIVE:
            attr = (attr < 0) * attr
            scaling_factor = -self._calculate_safe_scaling_factor(np.abs(attr))
        elif self.sign == AttributationVisualizationSign.ABSOLUTE_VALUE:
            attr = np.abs(attr)
            scaling_factor = self._calculate_safe_scaling_factor(attr)
        else:
            raise EnumValueNotFound(self.sign, AttributationVisualizationSign)
        attr_norm = self._normalize(attr, scaling_factor)
        return unflatten(obs_space, attr_norm)

    def _calculate_safe_scaling_factor(self, attr: AttributationLike) -> float:
        sorted_vals = np.sort(attr.flatten())
        cum_sums = np.cumsum(sorted_vals)
        threshold_id = np.where(
            cum_sums >= cum_sums[-1] * 0.01 * self.outlier_percentile
        )[0][0]
        return sorted_vals[threshold_id]

    def _normalize(
        self, attr: AttributationLike, scaling_factor: float
    ) -> AttributationLike:
        if abs(scaling_factor) < 1e-5:
            return np.clip(attr, -1, 1)
        attr_norm = attr / scaling_factor
        return np.clip(attr_norm, -1, 1)


class AttributationTarget(IntEnum):
    # The target equals to the actions picked by an agent
    PICKED = 0
    # The target chosen as argmax from action distribution
    HIGHEST = 1
    # 3 highest targets (calculated in the same way as `HIGHEST`)
    TOP3 = 3
    # 5 highest targets (calculated in the same way as `HIGHEST`)
    TOP5 = 5
    # All available targets
    ALL = sys.maxsize


@dataclass
class TimestepAttributationBatch:
    inputs: torch.Tensor
    baselines: torch.Tensor
    target: torch.Tensor
    timestep: Timestep


class AttributationTrajectoryIterator(abc.Iterator):
    def __init__(
        self,
        trajectory: Trajectory,
        model: Model,
        baseline: Optional[BaselineBuilder] = None,
        target: AttributationTarget = AttributationTarget.PICKED,
    ):
        self.trajectory = trajectory
        self.model = model
        self.baseline = baseline
        self.target = target
        self._it = iter(self.trajectory)

    def __next__(self) -> TimestepAttributationBatch:
        try:
            timestep = next(self._it)
        except StopIteration:
            raise StopIteration

        inputs = self.model.flatten_obs(timestep.obs)

        if self.baseline is not None:
            baselines = self.baseline(inputs)
        else:
            baselines = np.zeros_like(inputs)

        if self.target == AttributationTarget.PICKED:
            target = timestep.action
        else:
            raise NotImplementedError

        inputs = torch.tensor(inputs, device=self.model.input_device()).unsqueeze(dim=0)
        baselines = torch.tensor(
            baselines, device=self.model.output_device()
        ).unsqueeze(dim=0)
        target = torch.tensor(target, device=self.model.output_device())

        if isinstance(self.model.action_space(), gym.spaces.Discrete):
            # Discrete action space requires no modification to input signals
            pass
        elif isinstance(
            self.model.action_space(), (gym.spaces.MultiDiscrete, gym.spaces.Tuple)
        ):
            if isinstance(self.model.action_space(), gym.spaces.MultiDiscrete):
                # With MultiDiscrete action space we use batch dim to calculate
                # attributations for each action in this space
                subs = self.model.action_space().nvec
            else:
                # With Tuple action space we use batch dim to calculate attributations
                # for each subspace in the action space. The overall mechanism is very
                # similar to MultiDiscrete case, but each subspace might potentially be
                # a different space.
                subs = [space.n for space in self.model.action_space().spaces]

            num_subs = len(subs)

            # We are repeating the batch dim by the number of subs (actions or spaces)
            inputs = _extend_batch_dim(inputs, num_subs)
            baselines = _extend_batch_dim(baselines, num_subs)

            # The action is stored in a rollout as a vector of either (a1, a2, ...) or
            # (s1, s2, ...). However, in the model output, it is concatenated and
            # flattened to a single dimension. To compensate for that, we add an offset
            # from the beginning of the flattened vector to each action.
            # Example:
            # action_space = MultiDiscrete([4, 2])
            # rollout_action = np.array([1, 0])
            # model_action = [1, 0] + [0, 4] = [1, 4]
            offset = torch.tensor(subs).roll(1).to(device=self.model.output_device())
            offset[0] = 0
            target = target + offset
        else:
            raise ActionSpaceNotSupported(self.model.action_space())

        return TimestepAttributationBatch(
            inputs=inputs, baselines=baselines, target=target, timestep=timestep,
        )


# TODO attribute_rollout method


def attribute_trajectory(
    trajectory_it: AttributationTrajectoryIterator,
    model: Model,
    processor: Optional[AttributationProcessor] = None,
) -> Trajectory:
    algo = IntegratedGradients(model)
    timesteps = []
    for batch in trajectory_it:
        raw_attributation = algo.attribute(
            batch.inputs, baselines=batch.baselines, target=batch.target
        )

        if isinstance(model.action_space(), gym.spaces.Discrete):
            attributation = DiscreteActionAttributation(
                _convert_to_original_dimensions(model.obs_space(), raw_attributation)
            )
        elif isinstance(
            model.action_space(), (gym.spaces.MultiDiscrete, gym.spaces.Tuple)
        ):
            if isinstance(model.action_space(), gym.spaces.MultiDiscrete):
                # Using partial shouldn't be needed here obviously, but PyCharm
                # unexpectedly complains that cls is not callable at the instantiation
                # line...
                cls = partial(MultiDiscreteActionAttributation)
            else:
                cls = partial(TupleActionAttributation)
            attributation = cls(
                [
                    _convert_to_original_dimensions(
                        # Create a dummy batch dim, which is lost with this type of
                        # for-loop iteration
                        model.obs_space(),
                        action_attributation.unsqueeze(0),
                    )
                    for action_attributation in raw_attributation
                ]
            )
        else:
            raise ActionSpaceNotSupported(model.action_space())

        if processor is not None:
            attributation = processor.transform(attributation)

        timesteps.append(replace(batch.timestep, attributations=attributation))

    return Trajectory(timesteps)


def _extend_batch_dim(t: torch.Tensor, new_batch_dim: int) -> torch.Tensor:
    """
    Given a tensor `t` of shape [B x D1 x D2 x ...] we output the same tensor repeated
    along the batch dimension ([new_batch_dim x D1 x D2 x ...]).
    """
    num_non_batch_dims = len(t.shape[1:])
    repeat_shape = (new_batch_dim, *(1 for _ in range(num_non_batch_dims)))
    return t.repeat(repeat_shape)


def _convert_to_original_dimensions(
    obs_space: gym.Space, data: torch.Tensor
) -> ObsLike:
    if hasattr(obs_space, "original_space"):
        original_obs_space = obs_space.original_space
    else:
        original_obs_space = obs_space
    return unflatten(
        original_obs_space,
        flatten(
            original_obs_space,
            convert_to_non_torch_type(
                restore_original_dimensions(data, obs_space, tensorlib="torch")
            ),
        ),
    )
