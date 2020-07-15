import sys
from collections import abc
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Optional

import numpy as np
import torch
from captum.attr import IntegratedGradients

from rld.model import Model
from rld.processors import ObsPreprocessor, NoObsPreprocessor, AttributionPostprocessor
from rld.rollout import Trajectory, Timestep
from rld.typing import BaselineBuilder


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
class TimestepBatch:
    obs: torch.Tensor
    baseline: torch.Tensor
    target: torch.Tensor
    origin: Timestep


class AttributationTrajectoryIterator(abc.Iterator):
    def __init__(
        self,
        trajectory: Trajectory,
        model: Model,
        obs_preprocessor: Optional[ObsPreprocessor] = None,
        attr_postprocessor: Optional[AttributionPostprocessor] = None,
        baseline: Optional[BaselineBuilder] = None,
        target: AttributationTarget = AttributationTarget.PICKED,
    ):
        self.trajectory = trajectory
        self.model = model
        self.obs_preprocessor = (
            obs_preprocessor if obs_preprocessor is not None else NoObsPreprocessor()
        )
        self.attr_postprocessor = attr_postprocessor
        self.baseline = baseline
        self.target = target
        self._it = iter(self.trajectory)

    def __next__(self) -> TimestepBatch:
        try:
            timestep = next(self._it)
        except StopIteration:
            raise StopIteration

        obs = self.obs_preprocessor.transform(timestep.obs)

        if self.baseline is not None:
            baseline = self.baseline()
        else:
            baseline = np.zeros_like(obs)

        if self.target == AttributationTarget.PICKED:
            target = timestep.action
        else:
            raise NotImplementedError

        obs = torch.tensor(obs, device=self.model.input_device()).unsqueeze(dim=0)
        baseline = torch.tensor(baseline, device=self.model.output_device()).unsqueeze(
            dim=0
        )
        target = torch.tensor(target, device=self.model.output_device()).unsqueeze(
            dim=0
        )

        # Temporarily use only first sub-action in case of MultiDiscrete action space
        if target.size(1) > 1:
            target = target[:, 0]

        return TimestepBatch(
            obs=obs, baseline=baseline, target=target, origin=timestep,
        )


# TODO attribute_rollout method


def attribute_trajectory(
    trajectory_it: AttributationTrajectoryIterator, model: Model,
) -> Trajectory:
    algo = IntegratedGradients(model)
    timesteps = []
    for timestep in trajectory_it:
        attr = algo.attribute(
            timestep.obs, baselines=timestep.baseline, target=timestep.target
        )
        final_attr = trajectory_it.attr_postprocessor.transform(attr)
        # TODO Flatten batch dim
        timesteps.append(replace(timestep.origin, attributations=final_attr))

    return Trajectory(timesteps)
