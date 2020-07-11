import unittest
from pathlib import Path

import numpy as np

from rld.rollout import RayRollout


class TestRollout(unittest.TestCase):
    def test_ray_rollout(self):
        rollout = RayRollout(
            Path(__file__).parent / "resources" / "cartpole.ray_rollout"
        )
        self.assertEqual(rollout.num_episodes, 5)
        for trajectory in rollout:
            self.assertGreater(len(trajectory), 0)

            self.assertIsInstance(trajectory.obs, np.ndarray)
            self.assertIsInstance(trajectory.action, np.ndarray)
            self.assertIsInstance(trajectory.reward, np.ndarray)
            self.assertIsInstance(trajectory.done, np.ndarray)
            self.assertIsInstance(trajectory.info, list)

            for timestep in trajectory:
                self.assertIsInstance(timestep.obs, np.ndarray)
                self.assertIsInstance(timestep.action, (int, np.ndarray))
                self.assertIsInstance(timestep.reward, float)
                self.assertIsInstance(timestep.done, bool)
                self.assertIsInstance(timestep.info, dict)
