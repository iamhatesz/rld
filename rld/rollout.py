from __future__ import annotations

import shelve
from abc import ABC
from collections import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

from rld.typing import (
    ObsLike,
    is_obs_dict,
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
            obs=_index_obs(self.obs, item),
            action=_index_action(self.action, item),
            reward=float(self.reward[item]),
            done=bool(self.done[item]),
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
            obs=_stack_obs([ts[0] for ts in trajectory]),
            action=np.stack([ts[1] for ts in trajectory], axis=_BATCH_DIM),
            reward=np.stack([float(ts[3]) for ts in trajectory], axis=_BATCH_DIM),
            done=np.stack([bool(ts[4]) for ts in trajectory], axis=_BATCH_DIM),
            info=[ts[5] if len(ts) == 6 else {} for ts in trajectory],
        )


def _stack_obs(obss: Sequence[ObsLike]) -> ObsBatchLike:
    if is_obs_dict(obss[0]):
        size = len(obss)
        new_obs = {
            k: np.empty_like(v, shape=(size, *v.shape)) for k, v in obss[0].items()
        }
        for idx, obs in enumerate(obss):
            for key, value in obs.items():
                new_obs[key][idx] = value
        return new_obs
    return np.stack(obss, axis=_BATCH_DIM)


def _index_obs(obss: ObsBatchLike, index: int) -> ObsLike:
    if is_obs_dict(obss):
        return {k: v[index] for k, v in obss.items()}
    return obss[index]


def _index_action(actions: ActionBatchLike, index: int) -> ActionLike:
    if np.isscalar(actions[index]):
        return int(actions[index])
    return actions[index]
