import json
import shelve

import ray
import torch
from captum.attr import IntegratedGradients
from flask import Flask, render_template
from ray.rllib.agents.ppo import PPOTrainer

from rld.wrappers import RayModelWrapper

app = Flask(__name__, template_folder="ui", static_folder="ui/static")


@app.route("/")
def index():
    def model_from_ray_checkpoint(params_path: str, checkpoint_path: str):
        with open(params_path, "r") as f:
            params = json.load(f)

        ray.init(local_mode=True)
        trainer = PPOTrainer(config=params)
        trainer.restore(checkpoint_path)
        policy = trainer.get_policy()
        model = RayModelWrapper(policy.model)
        model.eval()
        ray.shutdown()
        return model

    def get_episode():
        rollout_path = r"/tmp/cartpole-evaluation"
        rollout = shelve.open(rollout_path)
        episode = rollout["0"]

        params_path = r"/tmp/cartpole/PPO/PPO_CartPole-v1_0_2020-07-09_15-32-27xjhmsljy/params.json"
        checkpoint_path = r"/tmp/cartpole/PPO/PPO_CartPole-v1_0_2020-07-09_15-32-27xjhmsljy/checkpoint_7/checkpoint-7"
        model = model_from_ray_checkpoint(params_path, checkpoint_path)
        ig = IntegratedGradients(model)

        baseline = torch.tensor([[0.0, 0.0, 0.0, 0.0]]).to("cuda")

        data = []
        for obs, action, _, _, _ in episode:
            input = torch.tensor(obs).to(device="cuda")
            attributions = ig.attribute(input, baseline, target=int(action))
            data.append(
                {
                    "obs": obs.tolist(),
                    "attribution": attributions.detach().cpu().tolist()[0],
                }
            )

        return data

    return render_template("index.html", episode=get_episode())


if __name__ == "__main__":
    app.run(debug=True)
