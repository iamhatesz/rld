from __future__ import annotations

import shelve
from abc import ABC
from collections import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Union, Optional

import numpy as np


@dataclass
class Timestep:
    obs: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    done: bool
    info: dict
    attributations: Optional[np.ndarray] = None

    @staticmethod
    def from_trajectory_index(
        obs: np.ndarray,
        action: Union[np.ndarray, np.ScalarType],
        reward: np.ScalarType,
        done: np.ScalarType,
        info: dict,
    ) -> Timestep:
        return Timestep(
            obs=obs,
            action=int(action) if np.isscalar(action) else action,
            reward=float(reward),
            done=bool(done),
            info=info,
        )


@dataclass
class Trajectory(abc.Iterable, abc.Sized):
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    info: List[dict]
    attributations: Optional[np.ndarray] = None

    def __getitem__(self, item: int) -> Timestep:
        return Timestep.from_trajectory_index(
            self.obs[item],
            self.action[item],
            self.reward[item],
            self.done[item],
            self.info[item],
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


class Rollout(ABC):
    @property
    def num_episodes(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Trajectory]:
        raise NotImplementedError


class RayRollout(Rollout):
    def __init__(self, rollout: Path):
        self.rollout = shelve.open(str(rollout))
        self.n = 0

    def __del__(self):
        self.rollout.close()

    @property
    def num_episodes(self) -> int:
        return self.rollout["num_episodes"]

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
            obs=np.stack([ts[0] for ts in trajectory], axis=0),
            action=np.stack([ts[1] for ts in trajectory], axis=0),
            reward=np.stack([float(ts[3]) for ts in trajectory], axis=0),
            done=np.stack([bool(ts[4]) for ts in trajectory], axis=0),
            info=[ts[5] if len(ts) == 6 else {} for ts in trajectory],
        )
