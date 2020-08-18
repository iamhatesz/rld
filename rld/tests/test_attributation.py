import unittest

import gym
import ray
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.ppo import PPOTrainer

from rld.attributation import (
    AttributationTrajectoryIterator,
    AttributationTarget,
    attribute_trajectory,
    AttributationNormalizer,
    AttributationNormalizationMode,
)
from rld.model import Model, RayModelWrapper
from rld.tests.resources.envs import ALL_ENVS, collect_rollout
from rld.tests.resources.models import ALL_MODELS
from rld.tests.resources.spaces import IMAGE_OBS_SPACE

ALL_TRAINERS = [
    PGTrainer,
    PPOTrainer,
]


class TestAttributation(unittest.TestCase):
    def _collect_rollout_and_validate_attributations(self, env: gym.Env, model: Model):
        obs_space = model.obs_space()
        rollout = collect_rollout(env, episodes=2, max_steps_per_episode=10)

        if obs_space is IMAGE_OBS_SPACE:
            obs_image_channel_dim = 2
        else:
            obs_image_channel_dim = None
        normalizer = AttributationNormalizer(
            obs_space=model.obs_space(),
            obs_image_channel_dim=obs_image_channel_dim,
            sign=AttributationNormalizationMode.ALL,
            outlier_percentile=5,
        )

        for trajectory in rollout:
            trajectory_it = AttributationTrajectoryIterator(
                trajectory, model=model, baseline=None, target=AttributationTarget.TOP3,
            )

            attr_trajectory = attribute_trajectory(
                trajectory_it, model, normalizer=normalizer
            )

            for timestep in attr_trajectory:
                self.assertIsNotNone(timestep.attributations)
                self.assertTrue(timestep.attributations.is_complied(obs_space))

    def test_model(self):
        for env_fn, model_fn in zip(ALL_ENVS, ALL_MODELS):
            with self.subTest(env=env_fn, model=model_fn):
                env = env_fn()
                model = model_fn()
                self._collect_rollout_and_validate_attributations(env, model)

    def test_ray_trainer_model(self):
        ray.init(local_mode=True)
        for trainer_fn in ALL_TRAINERS:
            for env_fn in ALL_ENVS:
                with self.subTest(trainer=trainer_fn, env=env_fn):
                    trainer = trainer_fn(config={"env": env_fn, "framework": "torch",})
                    model = RayModelWrapper(trainer.get_policy().model)
                    env = env_fn()
                    self._collect_rollout_and_validate_attributations(env, model)
        ray.shutdown()
