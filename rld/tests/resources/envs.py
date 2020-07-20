import random
from typing import Optional

import gym

from rld.rollout import Trajectory, Timestep, Rollout


class BaseEnv(gym.Env):
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
    observation_space = gym.spaces.Box(-1, 1, shape=(6,))
    action_space = gym.spaces.Discrete(4)


class BoxObsMultiDiscreteActionEnv(BaseEnv):
    observation_space = gym.spaces.Box(-1, 1, shape=(6,))
    action_space = gym.spaces.MultiDiscrete([4, 2])


class BoxObsTupleActionEnv(BaseEnv):
    observation_space = gym.spaces.Box(-1, 1, shape=(6,))
    action_space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(2)))


class ImageObsDiscreteActionEnv(BaseEnv):
    observation_space = gym.spaces.Box(-1, 1, shape=(84, 84, 4))
    action_space = gym.spaces.Discrete(4)


class ImageObsMultiDiscreteActionEnv(BaseEnv):
    observation_space = gym.spaces.Box(-1, 1, shape=(84, 84, 4))
    action_space = gym.spaces.MultiDiscrete([4, 2])


class ImageObsTupleActionEnv(BaseEnv):
    observation_space = gym.spaces.Box(-1, 1, shape=(84, 84, 4))
    action_space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(2)))


class DictObsDiscreteActionEnv(BaseEnv):
    observation_space = gym.spaces.Dict(
        a=gym.spaces.Box(-1, 1, (4, 6)), b=gym.spaces.Box(-1, 1, (2,))
    )
    action_space = gym.spaces.Discrete(4)


class DictObsMultiDiscreteActionEnv(BaseEnv):
    observation_space = gym.spaces.Dict(
        a=gym.spaces.Box(-1, 1, (4, 6)), b=gym.spaces.Box(-1, 1, (2,))
    )
    action_space = gym.spaces.MultiDiscrete([4, 2])


class DictObsTupleActionEnv(BaseEnv):
    observation_space = gym.spaces.Dict(
        a=gym.spaces.Box(-1, 1, (4, 6)), b=gym.spaces.Box(-1, 1, (2,))
    )
    action_space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(2)))


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
