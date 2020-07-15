import torch.nn as nn
from ray.rllib.models.preprocessors import Preprocessor

from rld.typing import ObsBatchLike, ObsBatchLikeStrict


class RayToCaptumModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def unwrapped(self):
        return self.model

    def forward(self, x):
        input_dict = {"obs": x, "obs_flat": x}
        state = None
        seq_lens = None
        return self.model(input_dict, state, seq_lens)[0]


# Rollout/Trajectory/Timestep/obs -> flatten -> Captum -> forward -> dictify -> output for flatten -> output for dictify


def preprocess_obs(
    obss: ObsBatchLike, preprocessor: Preprocessor
) -> ObsBatchLikeStrict:
    return [preprocessor.transform(obs) for obs in obss]
