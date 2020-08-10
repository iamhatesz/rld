from __future__ import annotations

import pickle
import shelve
from abc import ABC
from collections import abc, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Any, Sequence, Union, List, BinaryIO, Callable

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

    def map(
        self, fn: Callable[[AttributationLike], AttributationLike]
    ) -> ActionAttributation:
        raise NotImplementedError


@dataclass
class DiscreteActionAttributation(ActionAttributation):
    data: AttributationLike

    def is_complied(self, obs_space: gym.Space) -> bool:
        obs_space = remove_value_constraints_from_space(obs_space)
        return obs_space.contains(self.data)

    def action(self) -> AttributationLike:
        return self.data

    def map(
        self, fn: Callable[[AttributationLike], AttributationLike]
    ) -> ActionAttributation:
        return DiscreteActionAttributation(fn(self.data))


@dataclass
class MultiDiscreteActionAttributation(ActionAttributation):
    data: Sequence[AttributationLike]

    def is_complied(self, obs_space: gym.Space) -> bool:
        obs_space = remove_value_constraints_from_space(obs_space)
        return all(
            [obs_space.contains(self.action(i)) for i in range(self.num_actions())]
        )

    def map(
        self, fn: Callable[[AttributationLike], AttributationLike]
    ) -> ActionAttributation:
        return MultiDiscreteActionAttributation(
            [fn(self.action(i)) for i in range(self.num_actions())]
        )

    def num_actions(self) -> int:
        return len(self.data)

    def action(self, index: int) -> AttributationLike:
        return self.data[index]


@dataclass
class TupleActionAttributation(ActionAttributation):
    data: Sequence[AttributationLike]

    def is_complied(self, obs_space: gym.Space) -> bool:
        obs_space = remove_value_constraints_from_space(obs_space)
        return all(
            [obs_space.contains(self.space(i)) for i in range(self.num_spaces())]
        )

    def map(
        self, fn: Callable[[AttributationLike], AttributationLike]
    ) -> ActionAttributation:
        return TupleActionAttributation(
            [fn(self.space(i)) for i in range(self.num_spaces())]
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
        return iter(self.timesteps)

    def __len__(self) -> int:
        return len(self.timesteps)


@dataclass
class Rollout:
    trajectories: Sequence[Trajectory]

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self.trajectories)

    def __len__(self) -> int:
        return len(self.trajectories)


class RolloutReader:
    def __iter__(self) -> Iterator[Trajectory]:
        raise NotImplementedError


class FileRolloutReader(RolloutReader, Iterator[Trajectory]):
    def __init__(self, rollout_path: Path):
        self.rollout_path = rollout_path
        self._it: Optional[Iterator[Trajectory]] = None

    def __iter__(self) -> Iterator[Trajectory]:
        with open(str(self.rollout_path), "rb") as rollout_file:
            self._it = iter(pickle.load(rollout_file).trajectories)
        return self

    def __next__(self) -> Trajectory:
        return next(self._it)


class RayFileRolloutReader(RolloutReader, Iterator[Trajectory]):
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


class FileRolloutWriter(RolloutWriter):
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


def remove_value_constraints_from_space(obs_space: gym.Space):
    if isinstance(obs_space, gym.spaces.Box):
        return gym.spaces.Box(
            low=float("-inf"), high=float("+inf"), shape=obs_space.shape
        )
    elif isinstance(obs_space, gym.spaces.Dict):
        return gym.spaces.Dict(
            OrderedDict(
                [
                    (k, remove_value_constraints_from_space(space))
                    for k, space in obs_space.spaces.items()
                ]
            )
        )
    else:
        raise NotImplementedError


def remove_channel_dim_from_image_space(obs_space: gym.spaces.Box) -> gym.spaces.Box:
    return gym.spaces.Box(
        low=obs_space.low[..., 0],
        high=obs_space.high[..., 0],
        shape=obs_space.shape[:-1],
    )
