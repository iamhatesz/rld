import unittest
from importlib import resources
from pathlib import Path

import numpy as np

from rld.rollout import RayRolloutReader, FromMemoryRolloutReader
from rld.tests.resources.envs import collect_rollout, ALL_ENVS


class TestRolloutReader(unittest.TestCase):
    def test_from_memory(self):
        for env_fn in ALL_ENVS:
            with self.subTest(env=env_fn):
                rollout = FromMemoryRolloutReader(collect_rollout(env_fn()))
                for trajectory in rollout:
                    self.assertIsInstance(trajectory.timesteps, list)

    def test_ray_flat_observation(self):
        with resources.path(
            "rld.tests.resources.rollouts", "cartpole.ray_rollout"
        ) as f:
            rollout = RayRolloutReader(Path(f))
        for trajectory in rollout:
            self.assertIsInstance(trajectory.timesteps, list)

            for timestep in trajectory:
                self.assertIsInstance(timestep.obs, np.ndarray)
                self.assertIsInstance(timestep.action, int)
                self.assertIsInstance(timestep.reward, float)
                self.assertIsInstance(timestep.done, bool)
                self.assertIsInstance(timestep.info, dict)

    def test_ray_image_observation(self):
        with resources.path("rld.tests.resources.rollouts", "pong.ray_rollout") as f:
            rollout = RayRolloutReader(Path(f))
        for trajectory in rollout:
            self.assertIsInstance(trajectory.timesteps, list)

            for timestep in trajectory:
                self.assertIsInstance(timestep.obs, np.ndarray)
                self.assertIsInstance(timestep.action, int)
                self.assertIsInstance(timestep.reward, float)
                self.assertIsInstance(timestep.done, bool)
                self.assertIsInstance(timestep.info, dict)

    def test_ray_dict_observation(self):
        with resources.path(
            "rld.tests.resources.rollouts", "custom_env.ray_rollout"
        ) as f:
            rollout = RayRolloutReader(Path(f))
        for trajectory in rollout:
            self.assertIsInstance(trajectory.timesteps, list)

            for timestep in trajectory:
                self.assertIsInstance(timestep.obs, dict)
                self.assertIsInstance(timestep.action, np.ndarray)
                self.assertIsInstance(timestep.reward, float)
                self.assertIsInstance(timestep.done, bool)
                self.assertIsInstance(timestep.info, dict)
