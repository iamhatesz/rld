import unittest
from pathlib import Path

import ray
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models.preprocessors import get_preprocessor

from rld.attributation import (
    AttributationTrajectoryIterator,
    AttributationTarget,
    attribute_trajectory,
)
from rld.processors import RayObsPreprocessor, RayAttributionPostprocessor
from rld.rollout import RayRolloutReader, FromMemoryRolloutReader
from rld.tests.resources.envs import ALL_ENVS, collect_rollout
from rld.wrappers import RayModelWrapper


ALL_TRAINERS = [
    PGTrainer,
    PPOTrainer,
]


class TestAttributation(unittest.TestCase):
    def setUp(self) -> None:
        ray.init(local_mode=True)

    def tearDown(self) -> None:
        ray.shutdown()

    def test_ray_trainer_model(self):
        for trainer_fn in ALL_TRAINERS:
            for env_fn in ALL_ENVS:
                with self.subTest(trainer=trainer_fn, env=env_fn):
                    trainer = trainer_fn(config={"env": env_fn, "framework": "torch",})
                    model = RayModelWrapper(trainer.get_policy().model)
                    if hasattr(model.unwrapped().obs_space, "original_space"):
                        obs_space = model.unwrapped().obs_space.original_space
                    else:
                        obs_space = model.unwrapped().obs_space
                    preprocessor = get_preprocessor(obs_space)(obs_space)

                    rollout = FromMemoryRolloutReader(collect_rollout(env_fn()))

                    for trajectory in rollout:
                        trajectory_it = AttributationTrajectoryIterator(
                            trajectory,
                            model=model,
                            obs_preprocessor=RayObsPreprocessor(preprocessor),
                            attr_postprocessor=RayAttributionPostprocessor(
                                model.unwrapped()
                            ),
                            baseline=None,
                            target=AttributationTarget.PICKED,
                        )

                        attr_trajectory = attribute_trajectory(trajectory_it, model)

                        for timestep in attr_trajectory:
                            self.assertIsNotNone(timestep.attributations)
