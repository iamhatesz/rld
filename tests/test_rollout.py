import unittest
from pathlib import Path

from rld.rollout import RayRollout


class TestRollout(unittest.TestCase):
    def test_ray_rollout(self):
        rollout = RayRollout(
            Path(__file__).parent / "resources" / "cartpole.ray_rollout"
        )
        self.assertEqual(rollout.num_episodes, 5)
        for trajectory in rollout:
            self.assertGreater(len(trajectory), 0)
