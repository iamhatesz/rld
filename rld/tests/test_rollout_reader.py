import tempfile
import unittest
from pathlib import Path

import ray
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.rollout import rollout, RolloutSaver

from rld.rollout import RayRolloutReader, FromMemoryRolloutReader
from rld.tests.resources.envs import collect_rollout, ALL_ENVS

ALL_TRAINERS = [
    PGTrainer,
    PPOTrainer,
]


class TestRolloutReader(unittest.TestCase):
    def setUp(self) -> None:
        ray.init(local_mode=True)

    def tearDown(self) -> None:
        ray.shutdown()

    def test_from_memory(self):
        for env_fn in ALL_ENVS:
            with self.subTest(env=env_fn):
                rollout = FromMemoryRolloutReader(collect_rollout(env_fn()))
                for trajectory in rollout:
                    self.assertIsInstance(trajectory.timesteps, list)

    def test_from_ray_rollout(self):
        for trainer_fn in ALL_TRAINERS:
            for env_fn in ALL_ENVS:
                with self.subTest(trainer=trainer_fn, env=env_fn):
                    trainer = trainer_fn(config={"env": env_fn, "framework": "torch",})
                    with tempfile.TemporaryDirectory() as temp_dir:
                        rollout_file = Path(temp_dir) / "rollout"
                        rollout(
                            trainer,
                            env_fn,
                            num_steps=100,
                            num_episodes=10,
                            saver=RolloutSaver(outfile=str(rollout_file)),
                        )
                        rollout_reader = RayRolloutReader(rollout_file)
                        for trajectory in rollout_reader:
                            self.assertIsInstance(trajectory.timesteps, list)
