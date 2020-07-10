import json
import shelve

import ray
import torch
from captum.attr import IntegratedGradients
from ray.rllib.agents.ppo import PPOTrainer
from torch.nn import Module


class RayToCaptumModelWrapper(Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        input_dict = {"obs": x}
        state = None
        seq_lens = None
        return self.model(input_dict, state, seq_lens)[0]


if __name__ == "__main__":
    checkpoint_path = r"/tmp/cartpole/PPO/PPO_CartPole-v1_0_2020-07-09_15-32-27xjhmsljy/checkpoint_7/checkpoint-7"
    params_path = (
        r"/tmp/cartpole/PPO/PPO_CartPole-v1_0_2020-07-09_15-32-27xjhmsljy/params.json"
    )

    with open(params_path, "r") as f:
        params = json.load(f)

    ray.init()

    trainer = PPOTrainer(config=params)
    trainer.restore(checkpoint_path)
    policy = trainer.get_policy()
    model = RayToCaptumModelWrapper(policy.model)
    model.eval()

    baseline = torch.tensor([[0.0, 0.0, 0.0, 0.0]]).to("cuda")

    inputs = {
        "at_mid_going_left": [0.0, 0.0, 0.0, -2.0],
        "at_left_halfly_going_left": [-2.0, 0.0, -0.1, -2.0],
        "at_left_halfly_going_left_hard": [-2.0, 0.0, -0.1, -3.0],
    }
    selected_input = "at_left_halfly_going_left_hard"

    ig = IntegratedGradients(model)
    for input_name, input_value in inputs.items():
        input = torch.tensor([input_value]).to(device="cuda")
        print("Name:", input_name)
        print("Model prediction:", model(input))
        attributions_left = ig.attribute(input, baseline, target=0)
        attributions_right = ig.attribute(input, baseline, target=1)
        print("Input attributations for LEFT:", attributions_left)
        print("Input attributations for RIGHT:", attributions_right)
        print()

    rollout_path = r"/tmp/cartpole-evaluation"
    rollout = shelve.open(rollout_path)
    episode = rollout["0"]

    for obs, action, _, _, _ in episode:
        input = torch.tensor(obs).to(device="cuda")
        attributions = ig.attribute(input, baseline, target=int(action))
        print("Observation:", obs, "Action:", action)
        print("Attributation:", attributions)
        print()
