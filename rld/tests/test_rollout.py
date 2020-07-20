import tempfile
import unittest
from pathlib import Path

import numpy as np
import ray
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.rollout import rollout, RolloutSaver

from rld.rollout import (
    RayRolloutReader,
    FromMemoryRolloutReader,
    ToFileRolloutWriter,
    FromFileRolloutReader,
)
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

    def test_memory_rollouter_reader(self):
        for env_fn in ALL_ENVS:
            with self.subTest(env=env_fn):
                rollout = FromMemoryRolloutReader(collect_rollout(env_fn()))
                for trajectory in rollout:
                    self.assertIsInstance(trajectory.timesteps, list)

    def test_file_rollout_reader_writer(self):
        for env_fn in ALL_ENVS:
            rollout = collect_rollout(env_fn())
            with tempfile.TemporaryDirectory() as temp_dir:
                rollout_file = Path(temp_dir) / "rollout"

                with ToFileRolloutWriter(rollout_file) as writer:
                    for trajectory in rollout.trajectories:
                        writer.write(trajectory)

                reader = FromFileRolloutReader(rollout_file)
                for trajectory_l, trajectory_r in zip(rollout.trajectories, reader):
                    for ts_l, ts_r in zip(trajectory_l, trajectory_r):
                        np.testing.assert_equal(ts_l.obs, ts_r.obs)
                        np.testing.assert_equal(ts_l.action, ts_r.action)
                        self.assertEqual(ts_l.reward, ts_r.reward)
                        self.assertEqual(ts_l.done, ts_r.done)
                        np.testing.assert_equal(ts_l.info, ts_r.info)
                        np.testing.assert_equal(
                            ts_l.attributations, ts_r.attributations
                        )

    def test_ray_rollout_reader(self):
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
