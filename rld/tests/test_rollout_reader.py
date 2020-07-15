import unittest
from pathlib import Path

import numpy as np

from rld.rollout import RayRolloutReader


class TestRolloutReader(unittest.TestCase):
    def test_ray_flat_observation(self):
        rollout = RayRolloutReader(
            Path(__file__).parent / "resources" / "rollouts" / "cartpole.ray_rollout"
        )
        for trajectory in rollout:
            self.assertIsInstance(trajectory.timesteps, list)

            for timestep in trajectory:
                self.assertIsInstance(timestep.obs, np.ndarray)
                self.assertIsInstance(timestep.action, int)
                self.assertIsInstance(timestep.reward, float)
                self.assertIsInstance(timestep.done, bool)
                self.assertIsInstance(timestep.info, dict)

    def test_ray_image_observation(self):
        rollout = RayRolloutReader(
            Path(__file__).parent / "resources" / "rollouts" / "pong.ray_rollout"
        )
        for trajectory in rollout:
            self.assertIsInstance(trajectory.timesteps, list)

            for timestep in trajectory:
                self.assertIsInstance(timestep.obs, np.ndarray)
                self.assertIsInstance(timestep.action, int)
                self.assertIsInstance(timestep.reward, float)
                self.assertIsInstance(timestep.done, bool)
                self.assertIsInstance(timestep.info, dict)

    def test_ray_dict_observation(self):
        rollout = RayRolloutReader(
            Path(__file__).parent / "resources" / "rollouts" / "custom_env.ray_rollout"
        )
        for trajectory in rollout:
            self.assertIsInstance(trajectory.timesteps, list)

            for timestep in trajectory:
                self.assertIsInstance(timestep.obs, dict)
                self.assertIsInstance(timestep.action, np.ndarray)
                self.assertIsInstance(timestep.reward, float)
                self.assertIsInstance(timestep.done, bool)
                self.assertIsInstance(timestep.info, dict)
