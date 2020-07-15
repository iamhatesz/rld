from __future__ import annotations

import shelve
from collections import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Any, Sequence

import numpy as np

from rld.typing import (
    ObsLike,
    RewardLike,
    DoneLike,
    ActionLike,
    InfoLike,
    AttributationLike,
)

_BATCH_DIM = 0


@dataclass
class Timestep:
    obs: ObsLike
    action: ActionLike
    reward: RewardLike
    done: DoneLike
    info: InfoLike
    attributations: Optional[AttributationLike] = None


@dataclass
class Trajectory(abc.Iterable, abc.Sized):
    timesteps: Sequence[Timestep]
    # obs: ObsBatchLike
    # action: ActionBatchLike
    # reward: RewardBatchLike
    # done: DoneBatchLike
    # info: InfoBatchLike
    # attributations: Optional[AttributationBatchLike] = None

    # Right now we only support single timestep retrieval
    def __getitem__(self, item: int) -> Timestep:
        return Timestep(
            obs=self.timesteps[item].obs,
            action=self.timesteps[item].action,
            reward=self.timesteps[item].reward,
            done=self.timesteps[item].done,
            info=self.timesteps[item].info,
            attributations=self.timesteps[item].attributations,
        )

    def __iter__(self) -> Iterator[Trajectory]:
        return TrajectoryIterator(self)

    def __len__(self) -> int:
        return len(self.info)


class TrajectoryIterator(abc.Iterator):
    def __init__(self, trajectory: Trajectory):
        self.trajectory = trajectory
        self.n = 0

    def __next__(self) -> Timestep:
        try:
            timestep = self.trajectory[self.n]
            self.n += 1
            return timestep
        except IndexError:
            raise StopIteration


@dataclass
class Rollout:
    trajectories: Sequence[Trajectory]


class RolloutReader:
    def __iter__(self) -> RolloutIterator:
        raise NotImplementedError


class RolloutIterator:
    def __next__(self) -> Trajectory:
        raise NotImplementedError


class RayRolloutReader(RolloutReader, RolloutIterator):
    def __init__(self, rollout: Path):
        self.rollout = shelve.open(str(rollout))
        self.n = 0

    def __del__(self):
        self.rollout.close()

    def __iter__(self) -> Iterator[Trajectory]:
        self.n = 0
        return self

    def __next__(self) -> Trajectory:
        try:
            trajectory = self.rollout[f"{self.n}"]
        except KeyError:
            raise StopIteration
        self.n += 1
        return Trajectory(
            timesteps=[
                Timestep(
                    obs=ts[0],
                    action=to_int_if_scalar(ts[1]),
                    reward=float(ts[3]),
                    done=bool(ts[4]),
                    info=ts[5] if len(ts) == 6 else {},
                )
                for ts in trajectory
            ]
        )


def to_int_if_scalar(value: Any) -> ActionLike:
    if np.isscalar(value):
        return int(value)
    return value
