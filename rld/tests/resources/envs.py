import random
from abc import ABC
from typing import Optional

import gym

from rld.rollout import Trajectory, Timestep, Rollout
from rld.tests.resources.spaces import (
    BOX_OBS_SPACE,
    IMAGE_OBS_SPACE,
    DICT_OBS_SPACE,
    DISCRETE_ACTION_SPACE,
    MULTI_DISCRETE_ACTION_SPACE,
    TUPLE_ACTION_SPACE,
)


class BaseEnv(gym.Env, ABC):
    def __init__(self, env_config: Optional[dict] = None):
        super().__init__()
        self.env_config = env_config

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        reward = random.random()
        done = reward > 0.5
        return self.observation_space.sample(), reward, done, {}

    def render(self, mode="human"):
        pass


class BoxObsDiscreteActionEnv(BaseEnv):
    observation_space = BOX_OBS_SPACE
    action_space = DISCRETE_ACTION_SPACE


class BoxObsMultiDiscreteActionEnv(BaseEnv):
    observation_space = BOX_OBS_SPACE
    action_space = MULTI_DISCRETE_ACTION_SPACE


class BoxObsTupleActionEnv(BaseEnv):
    observation_space = BOX_OBS_SPACE
    action_space = TUPLE_ACTION_SPACE


class ImageObsDiscreteActionEnv(BaseEnv):
    observation_space = IMAGE_OBS_SPACE
    action_space = DISCRETE_ACTION_SPACE


class ImageObsMultiDiscreteActionEnv(BaseEnv):
    observation_space = IMAGE_OBS_SPACE
    action_space = MULTI_DISCRETE_ACTION_SPACE


class ImageObsTupleActionEnv(BaseEnv):
    observation_space = IMAGE_OBS_SPACE
    action_space = TUPLE_ACTION_SPACE


class DictObsDiscreteActionEnv(BaseEnv):
    observation_space = DICT_OBS_SPACE
    action_space = DISCRETE_ACTION_SPACE


class DictObsMultiDiscreteActionEnv(BaseEnv):
    observation_space = DICT_OBS_SPACE
    action_space = MULTI_DISCRETE_ACTION_SPACE


class DictObsTupleActionEnv(BaseEnv):
    observation_space = DICT_OBS_SPACE
    action_space = TUPLE_ACTION_SPACE


ALL_ENVS = [
    BoxObsDiscreteActionEnv,
    BoxObsMultiDiscreteActionEnv,
    # BoxObsTupleActionEnv,
    ImageObsDiscreteActionEnv,
    ImageObsMultiDiscreteActionEnv,
    # ImageObsTupleActionEnv,
    DictObsDiscreteActionEnv,
    DictObsMultiDiscreteActionEnv,
    # DictObsTupleActionEnv,
]


def collect_trajectory(env: gym.Env, max_steps: int = 100) -> Trajectory:
    obs = env.reset()
    timesteps = []
    i = 0
    while i < max_steps:
        action = env.action_space.sample()
        new_obs, reward, done, info = env.step(action)
        timesteps.append(Timestep(obs, action, reward, done, info))
        obs = new_obs
        if done:
            break
    return Trajectory(timesteps)


def collect_rollout(
    env: gym.Env, episodes: int = 10, max_steps_per_episode: int = 100
) -> Rollout:
    trajectories = []
    for episode in range(episodes):
        trajectories.append(collect_trajectory(env, max_steps_per_episode))
    return Rollout(trajectories)
