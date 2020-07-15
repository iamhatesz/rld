from __future__ import annotations

import shelve
from abc import ABC
from collections import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Any, Sequence

import numpy as np

from rld.typing import (
    ObsLike,
    RewardLike,
    ObsBatchLike,
    RewardBatchLike,
    DoneLike,
    DoneBatchLike,
    ActionLike,
    InfoLike,
    ActionBatchLike,
    InfoBatchLike,
)

_BATCH_DIM = 0


@dataclass
class Timestep:
    obs: ObsLike
    action: ActionLike
    reward: RewardLike
    done: DoneLike
    info: InfoLike
    attributations: Optional[np.ndarray] = None


@dataclass
class Trajectory(abc.Iterable, abc.Sized):
    obs: ObsBatchLike
    action: ActionBatchLike
    reward: RewardBatchLike
    done: DoneBatchLike
    info: InfoBatchLike
    attributations: Optional[np.ndarray] = None

    # Right now we only support single timestep retrieval
    def __getitem__(self, item: int) -> Timestep:
        return Timestep(
            obs=self.obs[item],
            action=self.action[item],
            reward=self.reward[item],
            done=self.done[item],
            info=self.info[item],
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


Episode = Trajectory


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
            obs=[ts[0] for ts in trajectory],
            action=[to_int_if_scalar(ts[1]) for ts in trajectory],
            reward=[float(ts[3]) for ts in trajectory],
            done=[bool(ts[4]) for ts in trajectory],
            info=[ts[5] if len(ts) == 6 else {} for ts in trajectory],
        )


def to_int_if_scalar(value: Any) -> ActionLike:
    if np.isscalar(value):
        return int(value)
    return value
