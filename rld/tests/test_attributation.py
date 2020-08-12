import unittest

import gym
import ray
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.ppo import PPOTrainer

from rld.attributation import (
    AttributationTrajectoryIterator,
    AttributationTarget,
    attribute_trajectory,
)
from rld.model import Model
from rld.tests.resources.envs import ALL_ENVS, collect_rollout
from rld.tests.resources.models import ALL_MODELS
from rld.wrappers import RayModelWrapper

ALL_TRAINERS = [
    PGTrainer,
    PPOTrainer,
]


class TestAttributation(unittest.TestCase):
    def _collect_rollout_and_validate_attributations(self, env: gym.Env, model: Model):
        if isinstance(model, RayModelWrapper):
            obs_space = model.original_obs_space()
        else:
            obs_space = model.obs_space()

        rollout = collect_rollout(env, episodes=2, max_steps_per_episode=10)

        for trajectory in rollout:
            trajectory_it = AttributationTrajectoryIterator(
                trajectory, model=model, baseline=None, target=AttributationTarget.TOP3,
            )

            attr_trajectory = attribute_trajectory(trajectory_it, model)

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
