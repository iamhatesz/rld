import sys
from collections import abc
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Optional

import gym
import numpy as np
import torch
from captum.attr import IntegratedGradients
from gym.spaces import flatten, unflatten
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.utils.torch_ops import convert_to_non_torch_type

from rld.exception import ActionSpaceNotSupported
from rld.model import Model
from rld.processors import ObsPreprocessor, NoObsPreprocessor
from rld.rollout import (
    Trajectory,
    Timestep,
    DiscreteActionAttributation,
    MultiDiscreteActionAttributation,
)
from rld.typing import BaselineBuilder, ObsLike


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
        obs_preprocessor: Optional[ObsPreprocessor] = None,
        baseline: Optional[BaselineBuilder] = None,
        target: AttributationTarget = AttributationTarget.PICKED,
    ):
        self.trajectory = trajectory
        self.model = model
        self.obs_preprocessor = (
            obs_preprocessor if obs_preprocessor is not None else NoObsPreprocessor()
        )
        self.baseline = baseline
        self.target = target
        self._it = iter(self.trajectory)

    def __next__(self) -> TimestepAttributationBatch:
        try:
            timestep = next(self._it)
        except StopIteration:
            raise StopIteration

        inputs = self.obs_preprocessor.transform(timestep.obs)

        if self.baseline is not None:
            baselines = self.baseline()
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
        elif isinstance(self.model.action_space(), gym.spaces.MultiDiscrete):
            # With MultiDiscrete action space we use batch dim to calculate
            # attributations for each action in this space
            action_sizes = self.model.action_space().nvec
            num_actions = len(action_sizes)
            num_obs_dims = len(self.model.obs_space().shape)

            # We are repeating the batch dim by the number of actions in action space
            repeat_shape = (num_actions, *(1 for _ in range(num_obs_dims)))
            inputs = inputs.repeat(repeat_shape)
            baselines = baselines.repeat(repeat_shape)

            # The multi discrete action is stored in a rollout as a vector (a1, a2, ...)
            # However, in the model output, it is concatenated and flattened to a single
            # dimension. Thus, to each action we add an offset from the beginning of the
            # flattened vector.
            # Example:
            # action_space = MultiDiscrete([4, 2])
            # rollout_action = np.array([1, 0])
            # model_action = [1, 0] + [0, 4] = [1, 4]
            offset = torch.tensor(action_sizes, device=self.model.output_device()).roll(
                1
            )
            offset[0] = 0
            target = target + offset
        elif isinstance(self.model.action_space(), gym.spaces.Tuple):
            raise NotImplementedError
        else:
            raise ActionSpaceNotSupported(self.model.action_space())

        return TimestepAttributationBatch(
            inputs=inputs, baselines=baselines, target=target, timestep=timestep,
        )


# TODO attribute_rollout method


def attribute_trajectory(
    trajectory_it: AttributationTrajectoryIterator, model: Model,
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
        elif isinstance(model.action_space(), gym.spaces.MultiDiscrete):
            attributation = MultiDiscreteActionAttributation(
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
        elif isinstance(model.action_space(), gym.spaces.Tuple):
            raise NotImplementedError
        else:
            raise ActionSpaceNotSupported(model.action_space())

        timesteps.append(replace(batch.timestep, attributations=attributation))

    return Trajectory(timesteps)


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
