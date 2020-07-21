from __future__ import annotations

import pickle
import shelve
from abc import ABC
from collections import abc
from dataclasses import dataclass
from io import BufferedWriter
from pathlib import Path
from typing import Iterator, Optional, Any, Sequence, Union, List, BinaryIO

import gym
import numpy as np

from rld.typing import (
    ObsLike,
    RewardLike,
    DoneLike,
    ActionLike,
    InfoLike,
    AttributationLike,
)


@dataclass
class ActionAttributation(ABC):
    data: Union[AttributationLike, Sequence[AttributationLike]]

    def is_complied(self, obs_space: gym.Space) -> bool:
        raise NotImplementedError


@dataclass
class DiscreteActionAttributation(ActionAttributation):
    data: AttributationLike

    def is_complied(self, obs_space: gym.Space) -> bool:
        return obs_space.contains(self.data)

    def action(self) -> AttributationLike:
        return self.data


@dataclass
class MultiDiscreteActionAttributation(ActionAttributation):
    data: Sequence[AttributationLike]

    def is_complied(self, obs_space: gym.Space) -> bool:
        return all(
            [obs_space.contains(self.action(i)) for i in range(self.num_actions())]
        )

    def num_actions(self) -> int:
        return len(self.data)

    def action(self, index: int) -> AttributationLike:
        return self.data[index]


@dataclass
class TupleActionAttributation(ActionAttributation):
    data: Sequence[AttributationLike]

    def is_complied(self, obs_space: gym.Space) -> bool:
        return all(
            [obs_space.contains(self.space(i)) for i in range(self.num_spaces())]
        )

    def num_spaces(self) -> int:
        return len(self.data)

    def space(self, index: int) -> AttributationLike:
        return self.data[index]


@dataclass
class Timestep:
    obs: ObsLike
    action: ActionLike
    reward: RewardLike
    done: DoneLike
    info: InfoLike
    attributations: Optional[ActionAttributation] = None


@dataclass
class Trajectory(abc.Iterable, abc.Sized):
    timesteps: Sequence[Timestep]

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

    def __iter__(self) -> Iterator[Timestep]:
        return TrajectoryIterator(self)

    def __len__(self) -> int:
        return len(self.timesteps)


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

    def __len__(self) -> int:
        return len(self.trajectories)


class RolloutReader:
    def __iter__(self) -> RolloutIterator:
        raise NotImplementedError


class RolloutIterator:
    def __next__(self) -> Trajectory:
        raise NotImplementedError


class FromMemoryRolloutReader(RolloutReader, RolloutIterator):
    def __init__(self, rollout: Rollout):
        self.rollout = rollout
        self._it: Optional[Iterator[Trajectory]] = None

    def __iter__(self) -> RolloutIterator:
        self._it = iter(self.rollout.trajectories)
        return self

    def __next__(self) -> Trajectory:
        return next(self._it)


class FromFileRolloutReader(RolloutReader, RolloutIterator):
    def __init__(self, rollout_path: Path):
        self.rollout_path = rollout_path
        self._it: Optional[Iterator[Trajectory]] = None

    def __iter__(self) -> RolloutIterator:
        with open(str(self.rollout_path), "rb") as rollout_file:
            self._it = iter(pickle.load(rollout_file).trajectories)
        return self

    def __next__(self) -> Trajectory:
        return next(self._it)


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


class RolloutWriter:
    def write(self, trajectory: Trajectory):
        pass


class ToFileRolloutWriter(RolloutWriter):
    def __init__(self, rollout_path: Path):
        self.rollout_path = rollout_path
        self._rollout_file: Optional[BinaryIO] = None
        self._trajectories: List[Trajectory] = []

    def __enter__(self) -> RolloutWriter:
        self._rollout_file = open(str(self.rollout_path), "wb")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        self._rollout_file.close()

    def write(self, trajectory: Trajectory):
        self._trajectories.append(trajectory)

    def flush(self):
        self._rollout_file.write(pickle.dumps(Rollout(self._trajectories)))
