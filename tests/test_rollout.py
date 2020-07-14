import unittest
from pathlib import Path

import numpy as np

from rld.rollout import RayRollout


class TestRollout(unittest.TestCase):
    def test_ray_flat_observation(self):
        rollout = RayRollout(
            Path(__file__).parent / "resources" / "cartpole.ray_rollout"
        )
        self.assertGreater(rollout.num_episodes, 0)
        for trajectory in rollout:
            self.assertGreater(len(trajectory), 0)

            self.assertIsInstance(trajectory.obs, np.ndarray)
            self.assertIsInstance(trajectory.action, np.ndarray)
            self.assertIsInstance(trajectory.reward, np.ndarray)
            self.assertIsInstance(trajectory.done, np.ndarray)
            self.assertIsInstance(trajectory.info, list)

            for timestep in trajectory:
                self.assertIsInstance(timestep.obs, np.ndarray)
                self.assertIsInstance(timestep.action, int)
                self.assertIsInstance(timestep.reward, float)
                self.assertIsInstance(timestep.done, bool)
                self.assertIsInstance(timestep.info, dict)

    def test_ray_image_observation(self):
        rollout = RayRollout(Path(__file__).parent / "resources" / "pong.ray_rollout")
        self.assertGreater(rollout.num_episodes, 0)
        for trajectory in rollout:
            self.assertGreater(len(trajectory), 0)

            self.assertIsInstance(trajectory.obs, np.ndarray)
            self.assertIsInstance(trajectory.action, np.ndarray)
            self.assertIsInstance(trajectory.reward, np.ndarray)
            self.assertIsInstance(trajectory.done, np.ndarray)
            self.assertIsInstance(trajectory.info, list)

            for timestep in trajectory:
                self.assertIsInstance(timestep.obs, np.ndarray)
                self.assertIsInstance(timestep.action, int)
                self.assertIsInstance(timestep.reward, float)
                self.assertIsInstance(timestep.done, bool)
                self.assertIsInstance(timestep.info, dict)

    def test_ray_dict_observation(self):
        rollout = RayRollout(
            Path(__file__).parent / "resources" / "custom_env.ray_rollout"
        )
        self.assertGreater(rollout.num_episodes, 0)
        for trajectory in rollout:
            self.assertGreater(len(trajectory), 0)

            self.assertIsInstance(trajectory.obs, dict)
            self.assertIsInstance(trajectory.action, np.ndarray)
            self.assertIsInstance(trajectory.reward, np.ndarray)
            self.assertIsInstance(trajectory.done, np.ndarray)
            self.assertIsInstance(trajectory.info, list)

            for timestep in trajectory:
                self.assertIsInstance(timestep.obs, dict)
                self.assertIsInstance(timestep.action, np.ndarray)
                self.assertIsInstance(timestep.reward, float)
                self.assertIsInstance(timestep.done, bool)
                self.assertIsInstance(timestep.info, dict)
