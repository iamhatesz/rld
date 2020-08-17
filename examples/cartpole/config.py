import gym
import numpy as np
import ray
from ray.rllib.agents.ppo import PPOTrainer

from rld.attributation import AttributationTarget
from rld.config import Config
from rld.model import Model, RayModelWrapper


def get_model() -> Model:
    ray.init()
    trainer = PPOTrainer(config={"env": "CartPole-v1", "framework": "torch",})
    model = RayModelWrapper(trainer.get_policy().model)
    ray.shutdown()
    return model


def build_baseline_builder(obs_space: gym.Space):
    return lambda: np.zeros_like(obs_space.sample())


model = get_model()


config = Config(
    model=model,
    baseline=build_baseline_builder(model.obs_space()),
    target=AttributationTarget.PICKED,
)
